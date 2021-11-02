#pragma once

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include <primitives_fitting/Utils.h>
#include <Eigen/Core>

#define EPS 1.0e-8

namespace primitives_fitting {

namespace ransac {

/**
 * @brief base sampler class
 *
 * @tparam T
 */
template <typename T>
class Sampler {
public:
    /**
     * @brief pure virtual operator, which define the I/O of this sampler
     *
     * @param indices
     * @param sample_size
     * @return std::vector<T>
     */
    virtual std::vector<T> operator()(const std::vector<T> &indices, size_t sample_size) const = 0;
};

/**
 * @brief Extract a random sample of given sample_size from the input indices
 *
 * @tparam T
 */
template <typename T>
class RandomSampler : public Sampler<T> {
public:
    std::vector<T> operator()(const std::vector<T> &indices, size_t sample_size) const override {
        std::random_device rd;
        std::mt19937 rng(rd());
        const size_t num = indices.size();
        std::vector<T> indices_ = indices;

        for (size_t i = 0; i < sample_size; ++i) {
            std::swap(indices_[i], indices_[rng() % num]);
        }

        std::vector<T> sample(sample_size);
        for (int idx = 0; idx < sample_size; ++idx) {
            sample[idx] = indices_[idx];
        }

        return sample;
    }
};

/**
 * @brief base primitives model
 *
 */
class Model {
public:
    Eigen::VectorXd m_parameters;  // The parameters of the current model

    Model(const Eigen::VectorXd &parameters) : m_parameters(parameters) {}
    Model &operator=(const Model &model) {
        m_parameters = model.m_parameters;
        return *this;
    }

    Model() {}
};

/**
 * @brief the plane model is described as [a, b, c, d] => ax + by + cz + d = 0
 *
 */
class Plane : public Model {
public:
    Plane() : Model(Eigen::VectorXd(4)){};
    Plane(const Plane &model) { m_parameters = model.m_parameters; }
    Plane &operator=(const Plane &model) {
        m_parameters = model.m_parameters;
        return *this;
    }
};

/**
 * @brief the sphere is describe as [x, y, z, r], where the first three are center and the
 * last is radius
 *
 */
class Sphere : public Model {
public:
    Sphere() : Model(Eigen::VectorXd(4)){};
    Sphere(const Sphere &model) { m_parameters = model.m_parameters; }
    Sphere &operator=(const Sphere &model) {
        m_parameters = model.m_parameters;
        return *this;
    }
};

/**
 * @brief the cylinder is describe as [x, y, z, nx, ny, nz, r], where the first three are center
 * the second three are normal vector or called direction vector, the last one is radius
 *
 */
class Cylinder : public Model {
public:
    Cylinder() : Model(Eigen::VectorXd(7)){};
    Cylinder(const Cylinder &model) { m_parameters = model.m_parameters; }
    Cylinder &operator=(const Cylinder &model) {
        m_parameters = model.m_parameters;
        return *this;
    }
};

class ModelEstimator {
protected:
    ModelEstimator(int minimal_sample) : m_minimal_sample(minimal_sample) {}

    /**
     * @brief check whether number of input points meet the minimal requirement
     *
     * @param num
     * @return true
     * @return false
     */
    bool MinimalCheck(int num) const { return num >= m_minimal_sample ? true : false; }

public:
    /**
     * @brief fit model using least sample points
     *
     * @param points
     * @param model
     * @return true
     * @return false
     */
    virtual bool MinimalFit(const Eigen::Matrix3Xd &points, Model &model) const = 0;

    /**
     * @brief fit model using least square method
     *
     * @param points
     * @param model
     * @return true
     * @return false
     */
    virtual bool GeneralFit(const Eigen::Matrix3Xd &points, Model &model) const = 0;

    /**
     * @brief evaluate point distance to model to determine inlier&outlier
     *
     * @param query
     * @param model
     * @return double
     */
    virtual double CalcPointToModelDistance(const Eigen::Vector3d &query,
                                            const Model &model) const = 0;

public:
    int m_minimal_sample;
};

class PlaneEstimator : public ModelEstimator {
public:
    PlaneEstimator() : ModelEstimator(3) {}

    bool MinimalFit(const Eigen::Matrix3Xd &points, Model &model) const override {
        if (!MinimalCheck(points.cols())) {
            return false;
        }

        const Eigen::Vector3d e0 = points.col(1) - points.col(0);
        const Eigen::Vector3d e1 = points.col(2) - points.col(0);
        Eigen::Vector3d abc = e0.cross(e1);
        const double norm = abc.norm();
        // if the three points are co-linear, return invalid plane
        if (norm < EPS) {
            return false;
        }
        abc /= abc.norm();
        const double d = -abc.dot(points.col(0));
        model.m_parameters(0) = abc(0);
        model.m_parameters(1) = abc(1);
        model.m_parameters(2) = abc(2);
        model.m_parameters(3) = d;

        return true;
    }

    bool GeneralFit(const Eigen::Matrix3Xd &points, Model &model) const override {
        const size_t num = points.cols();
        if (!MinimalCheck(num)) {
            return false;
        }

        const Eigen::Vector3d mean = points.rowwise().mean();

        double xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
#pragma omp parallel for reduction(+ : xx, xy, xz, yy, yz, zz)
        for (size_t i = 0; i < num; ++i) {
            const Eigen::Vector3d residual = points.col(i) - mean;
            xx += residual(0) * residual(0);
            xy += residual(0) * residual(1);
            xz += residual(0) * residual(2);
            yy += residual(1) * residual(1);
            yz += residual(1) * residual(2);
            zz += residual(2) * residual(2);
        }

        const double det_x = yy * zz - yz * yz;
        const double det_y = xx * zz - xz * xz;
        const double det_z = xx * yy - xy * xy;

        Eigen::Vector3d abc;
        if (det_x > det_y && det_x > det_z) {
            abc = Eigen::Vector3d(det_x, xz * yz - xy * zz, xy * yz - xz * yy);
        } else if (det_y > det_z) {
            abc = Eigen::Vector3d(xz * yz - xy * zz, det_y, xy * xz - yz * xx);
        } else {
            abc = Eigen::Vector3d(xy * yz - xz * yy, xy * xz - yz * xx, det_z);
        }

        const double norm = abc.norm();
        if (norm < EPS) {
            return false;
        }

        abc /= norm;
        const double d = -abc.dot(mean);
        model.m_parameters = Eigen::Vector4d(abc(0), abc(1), abc(2), d);

        return true;
    }

    double CalcPointToModelDistance(const Eigen::Vector3d &query,
                                    const Model &model) const override {
        const Eigen::Vector4d p(query(0), query(1), query(2), 1);
        return std::abs(model.m_parameters.transpose() * p) / model.m_parameters.head<3>().norm();
    }
};

class SphereEstimator : public ModelEstimator {
private:
    bool ValidationCheck(const Eigen::Matrix3Xd &points) const {
        PlaneEstimator fit;
        Plane plane;
        const Eigen::Matrix3Xd temp = points(Eigen::all, {0, 1, 2});
        const bool ret = fit.MinimalFit(temp, plane);
        if (!ret) {
            return false;
        }
        return fit.CalcPointToModelDistance(points.col(3), plane) < EPS ? false : true;
    }

public:
    SphereEstimator() : ModelEstimator(4) {}

    bool MinimalFit(const Eigen::Matrix3Xd &points, Model &model) const override {
        if (!MinimalCheck(points.cols()) || !ValidationCheck(points)) {
            return false;
        }

        Eigen::Matrix4d det_mat;
        det_mat.setOnes(4, 4);
        for (size_t i = 0; i < 4; i++) {
            det_mat(i, 0) = points(0, i);
            det_mat(i, 1) = points(1, i);
            det_mat(i, 2) = points(2, i);
        }
        const double M11 = det_mat.determinant();

        for (size_t i = 0; i < 4; i++) {
            det_mat(i, 0) = points.col(i).transpose() * points.col(i);
            det_mat(i, 1) = points(1, i);
            det_mat(i, 2) = points(2, i);
        }
        const double M12 = det_mat.determinant();

        for (size_t i = 0; i < 4; i++) {
            det_mat(i, 0) = points.col(i).transpose() * points.col(i);
            det_mat(i, 1) = points(0, i);
            det_mat(i, 2) = points(2, i);
        }
        const double M13 = det_mat.determinant();

        for (size_t i = 0; i < 4; i++) {
            det_mat(i, 0) = points.col(i).transpose() * points.col(i);
            det_mat(i, 1) = points(0, i);
            det_mat(i, 2) = points(1, i);
        }
        const double M14 = det_mat.determinant();

        for (size_t i = 0; i < 4; i++) {
            det_mat(i, 0) = points.col(i).transpose() * points.col(i);
            det_mat(i, 1) = points(0, i);
            det_mat(i, 2) = points(1, i);
            det_mat(i, 3) = points(2, i);
        }
        const double M15 = det_mat.determinant();

        const Eigen::Vector3d center(0.5 * (M12 / M11), -0.5 * (M13 / M11), 0.5 * (M14 / M11));
        const double radius = std::sqrt(center.transpose() * center - (M15 / M11));
        model.m_parameters(0) = center(0);
        model.m_parameters(1) = center(1);
        model.m_parameters(2) = center(2);
        model.m_parameters(3) = radius;

        return true;
    }

    bool GeneralFit(const Eigen::Matrix3Xd &points, Model &model) const override {
        const size_t num = points.cols();
        if (!MinimalCheck(num)) {
            return false;
        }

        Eigen::Matrix<double, Eigen::Dynamic, 4> A;
        A.setOnes(num, 4);
        A.col(0) = points.row(0).transpose() * 2;
        A.col(1) = points.row(1).transpose() * 2;
        A.col(2) = points.row(2).transpose() * 2;

        Eigen::VectorXd b = (points.row(0).array().pow(2) + points.row(1).array().pow(2) +
                             points.row(2).array().pow(2))
                                .matrix();

        // TODO: dangerous when b is very large, which need large memory to compute v. should be
        // improved.
        const Eigen::Vector4d w = A.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b);
        const double radius = sqrt(w(0) * w(0) + w(1) * w(1) + w(2) * w(2) + w(3));
        model.m_parameters(0) = w(0);
        model.m_parameters(1) = w(1);
        model.m_parameters(2) = w(2);
        model.m_parameters(3) = radius;

        return true;
    }

    double CalcPointToModelDistance(const Eigen::Vector3d &query,
                                    const Model &model) const override {
        const Eigen::Vector3d center = model.m_parameters.head<3>();
        const double radius = model.m_parameters(3);
        const double d = (query - center).norm();

        if (d <= radius) {
            return radius - d;
        } else {
            return d - radius;
        }
    }
};

// TODO: the cylinder estimator currently is not very well, recommanded to use parallel
// ransac with large number of max iteration (more than 5000)
class CylinderEstimator : public ModelEstimator {
public:
    CylinderEstimator() : ModelEstimator(3) {}

    bool MinimalFit(const Eigen::Matrix3Xd &points, Model &model) const override {
        if (!MinimalCheck(points.cols())) {
            return false;
        }
        const Eigen::Vector3d point1 = points.col(0);
        const Eigen::Vector3d point2 = points.col(1);
        const Eigen::Vector3d point3 = points.col(2);

        const Eigen::Matrix4d transform = utils::CalcCoordinateTransform(point1, point2, point3);
        const Eigen::Matrix4d transform_inv = transform.inverse();

        const Eigen::Vector3d p1 =
            (transform_inv.block(0, 0, 3, 3) * point1 + transform_inv.block(0, 3, 3, 1))
                .transpose();
        const Eigen::Vector3d p2 =
            (transform_inv.block(0, 0, 3, 3) * point2 + transform_inv.block(0, 3, 3, 1))
                .transpose();
        const Eigen::Vector3d p3 =
            (transform_inv.block(0, 0, 3, 3) * point3 + transform_inv.block(0, 3, 3, 1))
                .transpose();

        // Colinear check
        Eigen::Matrix3d half_determinant;
        half_determinant << p1(0), p2(0), p3(0), p1(1), p2(1), p3(1), 1, 1, 1;

        const double flag = half_determinant.determinant();

        if (flag == 0) {
            return false;
        } else {
            Eigen::Matrix3d a;
            Eigen::Matrix3d b;
            Eigen::Matrix3d c;
            Eigen::Matrix3d d;

            a << p1(0), p1(1), 1, p2(0), p2(1), 1, p3(0), p3(1), 1;

            b << p1(0) * p1(0) + p1(1) * p1(1), p1(1), 1, p2(0) * p2(0) + p2(1) * p2(1), p2(1), 1,
                p3(0) * p3(0) + p3(1) * p3(1), p3(1), 1;

            c << p1(0) * p1(0) + p1(1) * p1(1), p1(0), 1, p2(0) * p2(0) + p2(1) * p2(1), p2(0), 1,
                p3(0) * p3(0) + p3(1) * p3(1), p3(0), 1;

            d << p1(0) * p1(0) + p1(1) * p1(1), p1(0), p1(1), p2(0) * p2(0) + p2(1) * p2(1), p2(0),
                p2(1), p3(0) * p3(0) + p3(1) * p3(1), p3(0), p3(1);

            const double A = a.determinant();
            const double B = -b.determinant();
            const double C = c.determinant();
            const double D = -d.determinant();

            double x = -B / (2 * A);
            double y = -C / (2 * A);
            double z = 0;
            const double r = sqrt((B * B + C * C - 4 * A * D) / (4 * A * A));

            Eigen::Vector3d point_plane(x, y, z);
            point_plane = transform.block(0, 0, 3, 3) * point_plane + transform.block(0, 3, 3, 1);

            const double nx = transform(0, 2);
            const double ny = transform(1, 2);
            const double nz = transform(2, 2);

            model.m_parameters(0) = point_plane(0);
            model.m_parameters(1) = point_plane(1);
            model.m_parameters(2) = point_plane(2);
            model.m_parameters(3) = nx;
            model.m_parameters(4) = ny;
            model.m_parameters(5) = nz;
            model.m_parameters(6) = r;

            return true;
        }
    }

    /**
     * @brief the general fit of clinder model is not implemented yet
     *
     * @param points
     * @param model
     * @return true
     * @return false
     */
    bool GeneralFit(const Eigen::Matrix3Xd &points, Model &model) const override {
        // TODO: implement a better cylinder fit method
        return true;
    }

    double CalcPointToModelDistance(const Eigen::Vector3d &query,
                                    const Model &model) const override {
        const Eigen::Matrix<double, 7, 1> w = model.m_parameters;
        const Eigen::Vector3d n(w(3), w(4), w(5));
        const Eigen::Vector3d center(w(0), w(1), w(2));

        const Eigen::Vector3d ref(w(0) + w(3), w(1) + w(4), w(2) + w(5));
        double d = utils::CalcPoint2LineDistance(query, center, ref);

        return abs(d - w(6));
    }
};

template <class ModelEstimator, class Model, class Sampler>
class RANSAC {
public:
    RANSAC()
        : m_fitness(0)
        , m_inlier_rmse(0)
        , m_max_iteration(1000)
        , m_probability(0.99)
        , m_enable_parallel(false) {}

    /**
     * @brief set probability to find the best model
     *
     * @param probability
     */
    void SetProbability(double probability) { m_probability = probability; }

    /**
     * @brief set maximum iteration, usually used if using parallel ransac fitting
     *
     * @param num
     */
    void SetMaxIteration(size_t num) { m_max_iteration = num; }

    /**
     * @brief enable parallel ransac fitting
     *
     * @param flag
     */
    void SetParallel(bool flag) { m_enable_parallel = flag; }

    /**
     * @brief fit model with given parameters
     *
     * @param points
     * @param threshold
     * @param model
     * @param inlier_indices
     * @return true
     * @return false
     */
    bool FitModel(const std::vector<Eigen::Vector3d> &points, double threshold, Model &model,
                  std::vector<size_t> &inlier_indices) {
        Clear();
        const size_t num_points = points.size();
        if (num_points < m_estimator.m_minimal_sample) {
            std::cout << "can not fit model due to lack of points" << std::endl;
            return false;
        }

        if (m_enable_parallel) {
            return FitModelParallel(points, threshold, model, inlier_indices);
        } else {
            return FitModelVanilla(points, threshold, model, inlier_indices);
        }
    }

private:
    void Clear() {
        m_fitness = 0;
        m_inlier_rmse = 0;
    }

    /**
     * @brief refine model using general fitting of estimator, usually is least square method.
     *
     * @param points
     * @param threshold
     * @param model
     * @param inlier_indices
     * @return true
     * @return false
     */
    bool RefineModel(const std::vector<Eigen::Vector3d> &points, double threshold, Model &model,
                     std::vector<size_t> &inlier_indices) {
        inlier_indices.clear();
        for (size_t i = 0; i < points.size(); ++i) {
            const double d = m_estimator.CalcPointToModelDistance(points[i], model);
            if (d < threshold) {
                inlier_indices.emplace_back(i);
            }
        }

        // improve best model using general fitting
        Eigen::Matrix3Xd inlier_points;
        utils::GetMatrixByIndex(points, inlier_indices, inlier_points);

        return m_estimator.GeneralFit(inlier_points, model);
    }

    /**
     * @brief vanilla ransac fitting method, the iteration number is varying with inlier number in
     * each iteration
     *
     * @param points
     * @param threshold
     * @param model
     * @param inlier_indices
     * @return true
     * @return false
     */
    bool FitModelVanilla(const std::vector<Eigen::Vector3d> &points, double threshold, Model &model,
                         std::vector<size_t> &inlier_indices) {
        const size_t num_points = points.size();
        std::vector<size_t> indices_list(num_points);
        std::iota(std::begin(indices_list), std::end(indices_list), 0);

        Model best_model;
        size_t count = 0;
        size_t current_iteration = m_max_iteration;
        for (size_t i = 0; i < current_iteration; ++i) {
            const std::vector<size_t> sample_indices =
                m_sampler(indices_list, m_estimator.m_minimal_sample);
            Eigen::Matrix3Xd sample;
            utils::GetMatrixByIndex(points, sample_indices, sample);

            const bool ret = m_estimator.MinimalFit(sample, model);
            if (!ret) {
                continue;
            }

            const auto result = EvaluateModel(points, threshold, model);
            double fitness = std::get<0>(result);
            double inlier_rmse = std::get<1>(result);

            // update model if satisfy both fitness and rmse check
            if (fitness > m_fitness || (fitness == m_fitness && inlier_rmse < m_inlier_rmse)) {
                m_fitness = fitness;
                m_inlier_rmse = inlier_rmse;
                best_model = model;
                const size_t tmp =
                    log(1 - m_probability) / log(1 - pow(fitness, m_estimator.m_minimal_sample));
                if (tmp < 0) {
                    current_iteration = m_max_iteration;
                } else {
                    current_iteration = std::min(tmp, m_max_iteration);
                }
            }

            // break the loop if count larger than max iteration
            if (current_iteration > m_max_iteration) {
                break;
            }
            count++;
        }

        std::cout << "[vanilla ransac] find best model with " << 100 * m_fitness
                  << "% inliers and run " << count << " iterations" << std::endl;

        const bool ret = RefineModel(points, threshold, best_model, inlier_indices);
        model = best_model;
        return ret;
    }

    /**
     * @brief parallel ransac fitting method, the iteration number is fixed and use OpenMP to speed up.
     * Usually used if you want more accurate result.
     *
     * @param points
     * @param threshold
     * @param model
     * @param inlier_indices
     * @return true
     * @return false
     */
    bool FitModelParallel(const std::vector<Eigen::Vector3d> &points, double threshold,
                          Model &model, std::vector<size_t> &inlier_indices) {
        const size_t num_points = points.size();

        std::vector<size_t> indices_list(num_points);
        std::iota(std::begin(indices_list), std::end(indices_list), 0);

        std::vector<std::tuple<double, double, Model>> result_list;
#pragma omp parallel for shared(indices_list)
        for (size_t i = 0; i < m_max_iteration; ++i) {
            const std::vector<size_t> sample_indices =
                m_sampler(indices_list, m_estimator.m_minimal_sample);
            Eigen::Matrix3Xd sample;
            utils::GetMatrixByIndex(points, sample_indices, sample);

            Model model_trial;
            const bool ret = m_estimator.MinimalFit(sample, model_trial);
            if (!ret) {
                continue;
            }

            const auto result = EvaluateModel(points, threshold, model_trial);
            double fitness = std::get<0>(result);
            double inlier_rmse = std::get<1>(result);
#pragma omp critical
            { result_list.emplace_back(std::make_tuple(fitness, inlier_rmse, model_trial)); }
        }

        // selection best result from stored patch
        double max_fitness = 0;
        double min_inlier_rmse = 1.0e+10;
        auto best_result =
            *std::max_element(result_list.begin(), result_list.end(),
                              [](const std::tuple<double, double, Model> &result1,
                                 const std::tuple<double, double, Model> &result2) {
                                  const double fitness1 = std::get<0>(result1);
                                  const double fitness2 = std::get<0>(result2);
                                  const double inlier_rmse1 = std::get<1>(result1);
                                  const double inlier_rmse2 = std::get<1>(result2);
                                  return (fitness2 > fitness1 ||
                                          (fitness1 == fitness2 && inlier_rmse2 < inlier_rmse1));
                              });
        Model best_model;
        best_model = std::get<2>(best_result);

        std::cout << "[parallel ransac] find best model with " << 100 * std::get<0>(best_result)
                  << "% inliers" << std::endl;

        const bool ret = RefineModel(points, threshold, best_model, inlier_indices);
        model = best_model;

        return ret;
    }

    std::tuple<double, double> EvaluateModel(const std::vector<Eigen::Vector3d> &points,
                                             double threshold, const Model &model) {
        size_t inlier_num = 0;
        double error = 0;

        for (size_t idx = 0; idx < points.size(); ++idx) {
            const double distance = m_estimator.CalcPointToModelDistance(points[idx], model);

            if (distance < threshold) {
                error += distance;
                inlier_num++;
            }
        }

        double fitness;
        double inlier_rmse;

        if (inlier_num == 0) {
            fitness = 0;
            inlier_rmse = 1e+10;
        } else {
            fitness = (double)inlier_num / (double)points.size();
            inlier_rmse = error / std::sqrt((double)inlier_num);
        }

        return std::make_tuple(fitness, inlier_rmse);
    }

private:
    double m_probability;
    size_t m_max_iteration;
    double m_fitness;
    double m_inlier_rmse;
    Sampler m_sampler;
    ModelEstimator m_estimator;
    bool m_enable_parallel;
};

using RandomIndexSampler = RandomSampler<size_t>;
using RANSACPlane = RANSAC<PlaneEstimator, Plane, RandomIndexSampler>;
using RANSACShpere = RANSAC<SphereEstimator, Sphere, RandomIndexSampler>;
using RANSACCylinder = RANSAC<CylinderEstimator, Cylinder, RandomIndexSampler>;

}  // namespace ransac
}  // namespace primitives_fitting