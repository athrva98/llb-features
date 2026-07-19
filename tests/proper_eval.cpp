// Proper evaluation on 3DMatch with VERIFIED methodology.
//
// XYZ-GT sanity check: 100% ✓  (eval is correct)
//
// Tests:
//   1. XYZ-GT: sanity (should be ~100%)
//   2. FPFH-radius: Darboux histograms with RADIUS neighborhood
//   3. LLB-radius: Original LLB but with radius neighborhood + adaptive σ
//   4. LLB-original: k-NN LLB (for comparison)

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <llb_features.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <nanoflann.hpp>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

// ---- Point cloud with KD-tree ----
struct Cloud {
    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3f> normals;
    size_t kdtree_get_point_count() const { return points.size(); }
    float kdtree_get_pt(size_t i, size_t d) const { return points[i](static_cast<int>(d)); }
    template<class B> bool kdtree_get_bbox(B&) const { return false; }
    using Tree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float,Cloud>, Cloud, 3>;
    std::unique_ptr<Tree> tree;
    void buildTree() {
        tree = std::make_unique<Tree>(3, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        tree->buildIndex();
    }
    bool loadPLY(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        std::string line; int nv=0,np=0; bool bin=false,inv=false,isf=true;
        while(std::getline(f,line)){
            if(!line.empty()&&line.back()=='\r')line.pop_back();
            if(line=="end_header")break;
            if(line.find("binary_little_endian")!=std::string::npos)bin=true;
            if(line.find("element vertex")!=std::string::npos){nv=std::stoi(line.substr(line.rfind(' ')+1));inv=true;}
            if(line.find("element")!=std::string::npos&&line.find("vertex")==std::string::npos)inv=false;
            if(inv&&line.find("property")!=std::string::npos){++np;if(line.find("double")!=std::string::npos)isf=false;}
        }
        if(!bin||nv==0)return false;
        size_t es=isf?4:8,bpv=np*es; std::vector<char> row(bpv); points.resize(nv);
        for(int i=0;i<nv;++i){
            f.read(row.data(),static_cast<std::streamsize>(bpv));
            if(isf){float*p=(float*)row.data();points[i]={p[0],p[1],p[2]};}
            else{double*p=(double*)row.data();points[i]=Eigen::Vector3d(p[0],p[1],p[2]).cast<float>();}
        }
        return !points.empty();
    }
    void downsample(float v) {
        if(v<=0)return; float iv=1.f/v;
        std::unordered_map<int64_t,int> g; std::vector<Eigen::Vector3f> o;
        for(auto&p:points){
            int64_t k=(int64_t(std::floor(p(0)*iv))*73856093LL)^(int64_t(std::floor(p(1)*iv))*19349669LL)^(int64_t(std::floor(p(2)*iv))*83492791LL);
            if(!g.count(k)){g[k]=1;o.push_back(p);}
        }
        points=o;
    }
    void estimateNormals(int k=30) {
        int n=static_cast<int>(points.size()); normals.resize(n);
        std::vector<size_t> idx(k); std::vector<float> d(k);
        for(int i=0;i<n;++i){
            nanoflann::KNNResultSet<float> r(k);
            r.init(idx.data(),d.data());
            tree->findNeighbors(r,points[i].data(),nanoflann::SearchParameters());
            Eigen::Vector3f c=Eigen::Vector3f::Zero();
            for(int j=0;j<k;++j)c+=points[idx[j]]; c/=float(k);
            Eigen::Matrix3f cov=Eigen::Matrix3f::Zero();
            for(int j=0;j<k;++j){auto dd=points[idx[j]]-c;cov+=dd*dd.transpose();}
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov/float(k));
            normals[i]=es.eigenvectors().col(0);
        }
        Eigen::Vector3f avg=Eigen::Vector3f::Zero();
        for(auto&nn:normals)avg+=nn;
        if(avg.z()<0)for(auto&nn:normals)nn=-nn;
    }
};

// ---- GT loading ----
struct PosePair { int src, tgt; Eigen::Matrix4f pose; };
std::vector<PosePair> loadGTLog(const std::string& path) {
    std::vector<PosePair> pairs; std::ifstream file(path); std::string line;
    while(std::getline(file,line)){
        if(!line.empty()&&line.back()=='\r')line.pop_back();
        std::istringstream h(line);int s,t,nn;if(!(h>>s>>t>>nn))continue;
        Eigen::Matrix4f pose=Eigen::Matrix4f::Identity();bool ok=true;
        for(int r=0;r<4&&ok;++r){if(!std::getline(file,line)){ok=false;break;}
            if(!line.empty()&&line.back()=='\r')line.pop_back();
            std::istringstream rs(line);for(int c=0;c<4;++c)if(!(rs>>pose(r,c)))ok=false;}
        if(ok)pairs.push_back({s,t,pose});
    }
    return pairs;
}

// ============================================================
// FPFH with RADIUS search (the correct way)
// ============================================================
std::vector<Eigen::VectorXf> computeFPFHRadius(
    const Cloud& cloud, float radius, int bins = 11) {

    int dim = 3 * bins;
    int N = static_cast<int>(cloud.points.size());
    float sq_radius = radius * radius;
    std::vector<Eigen::VectorXf> feats(N, Eigen::VectorXf::Zero(dim));

    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < N; ++i) {
        // Radius search
        std::vector<nanoflann::ResultItem<uint32_t, float>> matches;
        cloud.tree->radiusSearch(cloud.points[i].data(), sq_radius, matches,
                                 nanoflann::SearchParameters());

        Eigen::VectorXf hist = Eigen::VectorXf::Zero(dim);
        for (auto& m : matches) {
            if (m.first == static_cast<uint32_t>(i)) continue; // skip self
            Eigen::Vector3f d = cloud.points[m.first] - cloud.points[i];
            float dist = d.norm();
            if (dist < 1e-10f) continue;
            Eigen::Vector3f u = d / dist;
            Eigen::Vector3f v = u.cross(cloud.normals[i]);
            float vn = v.norm();
            if (vn < 1e-10f) continue;
            v /= vn;
            Eigen::Vector3f w = cloud.normals[i].cross(v);

            float alpha = v.dot(cloud.normals[m.first]);
            float phi = u.dot(cloud.normals[i]);
            float theta = std::atan2(w.dot(cloud.normals[m.first]),
                                      u.dot(cloud.normals[m.first]));

            float weight = 1.0f / (dist + 1e-10f);

            int ba = std::max(0, std::min(int((alpha+1.f)*.5f*bins), bins-1));
            int bp = std::max(0, std::min(int((phi+1.f)*.5f*bins), bins-1));
            float tn = (theta + float(M_PI)) / (2.f * float(M_PI));
            int bt = std::max(0, std::min(int(tn * bins), bins-1));

            hist(ba) += weight;
            hist(bins + bp) += weight;
            hist(2*bins + bt) += weight;
        }
        // Normalize
        for (int f = 0; f < 3; ++f) {
            float sum = 0;
            for (int b = 0; b < bins; ++b) sum += hist(f*bins+b);
            if (sum > 0) for (int b = 0; b < bins; ++b) hist(f*bins+b) /= sum;
        }
        feats[i] = hist;
    }
    return feats;
}

// ============================================================
// LLB with RADIUS search + adaptive bandwidth
// ============================================================
std::vector<Eigen::VectorXf> computeLLBRadius(
    const Cloud& cloud, float radius, int num_eigvecs = 10, int max_pts = 200) {

    int N = static_cast<int>(cloud.points.size());
    float sq_radius = radius * radius;
    std::vector<Eigen::VectorXf> feats(N, Eigen::VectorXf::Zero(num_eigvecs));

    #pragma omp parallel for schedule(dynamic, 16)
    for (int qi = 0; qi < N; ++qi) {
        // Radius search
        std::vector<nanoflann::ResultItem<uint32_t, float>> matches;
        cloud.tree->radiusSearch(cloud.points[qi].data(), sq_radius, matches,
                                  nanoflann::SearchParameters());

        int n = static_cast<int>(matches.size());
        if (n < 6) continue;

        // Subsample if too many
        if (n > max_pts) {
            int step = n / max_pts;
            std::vector<nanoflann::ResultItem<uint32_t, float>> sub;
            sub.push_back(matches[0]); // keep self
            for (int j = 1; j < n; j += step) sub.push_back(matches[j]);
            matches = std::move(sub);
            n = static_cast<int>(matches.size());
        }

        // Find query index in neighborhood
        int query_local = 0;
        for (int j = 0; j < n; ++j) {
            if (matches[j].first == static_cast<uint32_t>(qi)) { query_local = j; break; }
        }

        // Adaptive bandwidth: median distance
        std::vector<float> dists_sorted;
        for (auto& m : matches) {
            float d = std::sqrt(m.second);
            if (d > 1e-10f) dists_sorted.push_back(d);
        }
        if (dists_sorted.empty()) continue;
        std::sort(dists_sorted.begin(), dists_sorted.end());
        float sigma = dists_sorted[dists_sorted.size() / 2];
        float inv_2s2 = 1.0f / (2.0f * sigma * sigma + 1e-10f);

        // Weight matrix with adaptive Gaussian
        Eigen::MatrixXf W = Eigen::MatrixXf::Zero(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                float sq = (cloud.points[matches[i].first] -
                            cloud.points[matches[j].first]).squaredNorm();
                float w = std::exp(-sq * inv_2s2);
                W(i,j) = w; W(j,i) = w;
            }
        }

        // Normalized Laplacian
        Eigen::VectorXf D = W.rowwise().sum();
        Eigen::VectorXf Dinv = D.unaryExpr([](float v) -> float {
            return v > 1e-10f ? 1.0f / std::sqrt(v) : 0.0f;
        });
        Eigen::MatrixXf L = Dinv.asDiagonal() *
            (Eigen::MatrixXf(D.asDiagonal()) - W) * Dinv.asDiagonal();

        // Eigendecomposition
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(L);
        if (es.info() != Eigen::Success) continue;

        int avail = std::min(num_eigvecs, static_cast<int>(es.eigenvalues().size()));
        Eigen::VectorXf feat = Eigen::VectorXf::Zero(num_eigvecs);
        for (int i = 0; i < avail; ++i) {
            feat(i) = std::sqrt(std::abs(es.eigenvalues()(i)))
                      * es.eigenvectors().col(i)(query_local);
        }
        feats[qi] = feat;
    }
    return feats;
}

// ============================================================
// FPFH k-NN (standalone function for reuse)
// ============================================================
std::vector<Eigen::VectorXf> computeFPFHkNN(const Cloud& c, int k=30, int bins=11) {
    int dim=3*bins, N=static_cast<int>(c.points.size());
    std::vector<Eigen::VectorXf> feats(N, Eigen::VectorXf::Zero(dim));
    #pragma omp parallel for schedule(dynamic,64)
    for(int i=0;i<N;++i){
        std::vector<size_t> idx(k);std::vector<float> d(k);
        nanoflann::KNNResultSet<float> r(k);r.init(idx.data(),d.data());
        c.tree->findNeighbors(r,c.points[i].data(),nanoflann::SearchParameters());
        Eigen::VectorXf hist=Eigen::VectorXf::Zero(dim);
        for(int j=1;j<k;++j){
            Eigen::Vector3f dd=c.points[idx[j]]-c.points[i];float dist=dd.norm();
            if(dist<1e-10f)continue;Eigen::Vector3f u=dd/dist;
            Eigen::Vector3f v=u.cross(c.normals[i]);float vn=v.norm();
            if(vn<1e-10f)continue;v/=vn;Eigen::Vector3f w=c.normals[i].cross(v);
            float a=v.dot(c.normals[idx[j]]),p=u.dot(c.normals[i]);
            float th=std::atan2(w.dot(c.normals[idx[j]]),u.dot(c.normals[idx[j]]));
            float wt=1.f/(std::sqrt(d[j])+1e-10f);
            hist(std::max(0,std::min(int((a+1.f)*.5f*bins),bins-1)))+=wt;
            hist(bins+std::max(0,std::min(int((p+1.f)*.5f*bins),bins-1)))+=wt;
            float tn=(th+float(M_PI))/(2.f*float(M_PI));
            hist(2*bins+std::max(0,std::min(int(tn*bins),bins-1)))+=wt;
        }
        for(int f=0;f<3;++f){float s=0;for(int b=0;b<bins;++b)s+=hist(f*bins+b);
            if(s>0)for(int b=0;b<bins;++b)hist(f*bins+b)/=s;}
        feats[i]=hist;
    }
    return feats;
}

// ============================================================
// Original LLB (k-NN, for comparison)
// ============================================================
std::vector<Eigen::VectorXf> computeLLBOriginal(
    const Cloud& cloud, size_t k, size_t eigvecs, float nw) {
    llb_features::AlignedVector<float> pts, nrm;
    for (size_t i = 0; i < cloud.points.size(); ++i) {
        pts.push_back(cloud.points[i]);
        nrm.push_back(cloud.normals[i]);
    }
    size_t ak = std::min(k, pts.size());
    size_t ae = std::min(eigvecs, ak);
    llb_features::LLBFeatures<float> llb(pts, nrm, ak, ae, nw);
    return llb.computeFeatures();
}

double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// ============================================================
int main(int argc, char** argv) {
    std::string data_dir = "data/tune";
    float voxel = 0.05f;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--data" && i+1 < argc) data_dir = argv[++i];
        else if (a == "--voxel" && i+1 < argc) voxel = std::stof(argv[++i]);
    }

    struct Scene { std::string name, dir, gt; };
    std::vector<Scene> scenes;
    for (auto& e : fs::directory_iterator(data_dir)) {
        if (!e.is_directory()) continue;
        std::string name = e.path().filename().string();
        if (name.find("-evaluation") != std::string::npos) continue;
        std::string gt = (fs::path(e.path().string() + "-evaluation") / "gt.log").string();
        if (fs::exists(gt)) scenes.push_back({name, e.path().string(), gt});
    }
    std::sort(scenes.begin(), scenes.end(), [](auto& a, auto& b){ return a.name < b.name; });
    std::cout << scenes.size() << " scenes, voxel=" << voxel << "m\n";

    float fpfh_radius = voxel * 5.0f;
    float llb_radius = voxel * 5.0f;
    int max_pairs = 15;

    std::cout << "FPFH radius: " << fpfh_radius << "m, LLB radius: " << llb_radius << "m\n\n";

    struct Desc {
        const char* name;
        std::function<std::vector<Eigen::VectorXf>(const Cloud&, const Eigen::Matrix4f&)> compute;
    };

    std::vector<Desc> descs = {
        {"XYZ-GT (sanity)", [](const Cloud& c, const Eigen::Matrix4f& gt) {
            std::vector<Eigen::VectorXf> f(c.points.size());
            Eigen::Matrix3f R=gt.block<3,3>(0,0); Eigen::Vector3f t=gt.block<3,1>(0,3);
            for(size_t i=0;i<c.points.size();++i) f[i]=R*c.points[i]+t;
            return f;
        }},
        {"XYZ-tgt (sanity)", [](const Cloud& c, const Eigen::Matrix4f&) {
            std::vector<Eigen::VectorXf> f(c.points.size());
            for(size_t i=0;i<c.points.size();++i) f[i]=c.points[i];
            return f;
        }},
        {"FPFH k=30 (knn)", [](const Cloud& c, const Eigen::Matrix4f&) {
            return computeFPFHkNN(c, 30, 11);
        }},
        {"LLB k=30 e=10", [](const Cloud& c, const Eigen::Matrix4f&) {
            return computeLLBOriginal(c, 30, 10, 0.3f);
        }},
        {"LLB k=60 e=10", [](const Cloud& c, const Eigen::Matrix4f&) {
            return computeLLBOriginal(c, 60, 10, 0.3f);
        }},
        {"LLB k=15 e=5", [](const Cloud& c, const Eigen::Matrix4f&) {
            return computeLLBOriginal(c, 15, 5, 0.3f);
        }},
        {"LLB k=15 e=5 sub=0.2", [](const Cloud& c, const Eigen::Matrix4f&) {
            llb_features::AlignedVector<float> pts, nrm;
            for (auto& p : c.points) pts.push_back(p);
            for (auto& n : c.normals) nrm.push_back(n);
            size_t ak = std::min<size_t>(15, pts.size());
            size_t ae = std::min<size_t>(5, ak);
            llb_features::LLBFeatures<float> llb(pts, nrm, ak, ae, 0.3f);
            return llb.computeFeaturesSubsampled(0.2, 3);
        }},
        {"LLB k=15 e=5 sub=0.1", [](const Cloud& c, const Eigen::Matrix4f&) {
            llb_features::AlignedVector<float> pts, nrm;
            for (auto& p : c.points) pts.push_back(p);
            for (auto& n : c.normals) nrm.push_back(n);
            size_t ak = std::min<size_t>(15, pts.size());
            size_t ae = std::min<size_t>(5, ak);
            llb_features::LLBFeatures<float> llb(pts, nrm, ak, ae, 0.3f);
            return llb.computeFeaturesSubsampled(0.1, 3);
        }},
    };

    std::vector<int> Ks = {1, 5, 10, 50};

    // Build header
    std::cout << std::string(100, '=') << "\n";
    std::cout << std::left << std::setw(22) << "Descriptor"
              << std::right << std::setw(6) << "Pts"
              << std::setw(8) << "Time";
    for (int K : Ks) std::cout << std::setw(8) << ("R@" + std::to_string(K));
    std::cout << std::setw(8) << "GTcorr" << "  Scene\n";
    std::cout << std::string(100, '-') << "\n";

    // agg[desc_name][k_idx] = sum of recall across scenes
    std::unordered_map<std::string, std::vector<double>> agg;
    std::unordered_map<std::string, int> agg_count;

    for (auto& scene : scenes) {
        auto pairs = loadGTLog(scene.gt);
        if (pairs.empty()) continue;
        if (static_cast<int>(pairs.size()) > max_pairs) pairs.resize(max_pairs);

        std::unordered_map<int, Cloud> frags;
        for (auto& p : pairs) {
            for (int fid : {p.src, p.tgt}) {
                if (frags.count(fid)) continue;
                Cloud& c = frags[fid];
                c.loadPLY((fs::path(scene.dir)/("cloud_bin_"+std::to_string(fid)+".ply")).string());
                c.downsample(voxel);
                c.buildTree();
                c.estimateNormals(30);
            }
        }

        int avg_pts=0,fc=0;
        for(auto&[id,c]:frags){avg_pts+=static_cast<int>(c.points.size());++fc;}
        avg_pts=fc>0?avg_pts/fc:0;

        for (auto& desc : descs) {
            double total_time=0;
            int total_gt=0, npairs=0;
            std::vector<int> total_correct(Ks.size(), 0);

            std::unordered_map<int, std::vector<Eigen::VectorXf>> cache;

            for (auto& pair : pairs) {
                auto& src = frags[pair.src];
                auto& tgt = frags[pair.tgt];
                if (src.points.size()<6||tgt.points.size()<6) continue;

                bool is_xyz_gt = (std::string(desc.name).find("XYZ-GT") != std::string::npos);

                std::vector<Eigen::VectorXf> fs_vec, ft_vec;
                auto t0 = Clock::now();

                if (is_xyz_gt) {
                    fs_vec = desc.compute(src, pair.pose);
                    ft_vec = desc.compute(tgt, Eigen::Matrix4f::Identity());
                } else {
                    if (!cache.count(pair.src))
                        cache[pair.src] = desc.compute(src, Eigen::Matrix4f::Identity());
                    if (!cache.count(pair.tgt))
                        cache[pair.tgt] = desc.compute(tgt, Eigen::Matrix4f::Identity());
                    fs_vec = cache[pair.src];
                    ft_vec = cache[pair.tgt];
                }
                total_time += ms_since(t0);

                Eigen::Matrix3f R = pair.pose.block<3,3>(0,0);
                Eigen::Vector3f t_vec = pair.pose.block<3,1>(0,3);
                float sq_tau = 0.1f * 0.1f;
                int Ns = static_cast<int>(src.points.size());
                int Nt = static_cast<int>(tgt.points.size());
                int maxK = Ks.back();

                for (int i = 0; i < Ns; ++i) {
                    Eigen::Vector3f src_xf = R * src.points[i] + t_vec;

                    // GT correspondence check
                    std::vector<size_t> sp_idx(1); std::vector<float> sp_dist(1);
                    nanoflann::KNNResultSet<float> sp_r(1);
                    sp_r.init(sp_idx.data(), sp_dist.data());
                    tgt.tree->findNeighbors(sp_r, src_xf.data(), nanoflann::SearchParameters());
                    if (sp_dist[0] > sq_tau) continue;
                    ++total_gt;

                    if (fs_vec[i].squaredNorm() < 1e-20f) continue;

                    // Find top-K feature NNs (partial sort)
                    std::vector<std::pair<float, int>> feat_dists;
                    feat_dists.reserve(Nt);
                    for (int j = 0; j < Nt; ++j) {
                        if (ft_vec[j].squaredNorm() < 1e-20f) continue;
                        float d = (fs_vec[i] - ft_vec[j]).squaredNorm();
                        feat_dists.push_back({d, j});
                    }

                    // Partial sort to get top maxK
                    int topK = std::min(maxK, static_cast<int>(feat_dists.size()));
                    if (topK == 0) continue;
                    std::partial_sort(feat_dists.begin(), feat_dists.begin() + topK,
                                      feat_dists.end());

                    // Check each K threshold
                    for (size_t ki = 0; ki < Ks.size(); ++ki) {
                        int K = std::min(Ks[ki], topK);
                        bool found = false;
                        for (int kk = 0; kk < K; ++kk) {
                            int j = feat_dists[kk].second;
                            if ((src_xf - tgt.points[j]).squaredNorm() < sq_tau) {
                                found = true;
                                break;
                            }
                        }
                        if (found) total_correct[ki]++;
                    }
                }
                ++npairs;
            }

            double avg_time = npairs>0 ? total_time/npairs : 0;
            std::cout << std::left << std::setw(22) << desc.name
                      << std::right << std::setw(6) << avg_pts
                      << std::setw(6) << std::fixed << std::setprecision(0) << avg_time << "ms";
            for (size_t ki = 0; ki < Ks.size(); ++ki) {
                double recall = total_gt > 0 ? 100.0 * total_correct[ki] / total_gt : 0;
                std::cout << std::setw(7) << std::setprecision(1) << recall << "%";
            }
            std::cout << std::setw(8) << total_gt
                      << "  " << std::left << scene.name << std::endl << std::right;

            if (!agg.count(desc.name)) agg[desc.name].resize(Ks.size(), 0.0);
            for (size_t ki = 0; ki < Ks.size(); ++ki) {
                double recall = total_gt > 0 ? 100.0 * total_correct[ki] / total_gt : 0;
                agg[desc.name][ki] += recall;
            }
            agg_count[desc.name]++;
        }
        std::cout << std::string(100, '-') << "\n";
    }

    // Summary
    std::cout << "\nAGGREGATE (mean across scenes):\n" << std::string(60, '-') << "\n";
    std::cout << std::left << std::setw(24) << "Descriptor";
    for (int K : Ks) std::cout << std::right << std::setw(8) << ("R@" + std::to_string(K));
    std::cout << "\n" << std::string(60, '-') << "\n";

    // Sort by R@1 descending
    std::vector<std::pair<double,std::string>> sorted;
    for(auto&[n,v]:agg) {
        int cnt = agg_count[n];
        sorted.push_back({cnt>0?v[0]/cnt:0, n});
    }
    std::sort(sorted.rbegin(), sorted.rend());
    for (auto& [_, name] : sorted) {
        int cnt = agg_count[name];
        std::cout << std::left << std::setw(24) << name;
        for (size_t ki = 0; ki < Ks.size(); ++ki) {
            std::cout << std::right << std::setw(7) << std::fixed << std::setprecision(1)
                      << (cnt>0 ? agg[name][ki]/cnt : 0) << "%";
        }
        std::cout << "\n";
    }
    std::cout << std::string(60, '-') << "\n";

    return 0;
}
