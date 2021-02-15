/**
 * @author Andrey Larionov
 */
#ifndef CQUMO_TANDEM_STATISTICS_H
#define CQUMO_TANDEM_STATISTICS_H

#include <vector>
#include <string>
#include <cmath>
#include "../Base.h"


namespace cqumo {

/**
 * Get unbiased variance estimation.
 * If no number of samples provided, returns biased estimation.
 * Otherwise, multiplies biased estimation `m2 - m1*m1` by `n/(n-1)`
 * to value unbiased estimation.
 *
 * @param m1 sample mean
 * @param m2 sample moment of order 2
 * @param n number of samples
 * @return `(m2 - m1^2) * (n/(n-1))`
 */
double getUnbiasedVariance(double m1, double m2, unsigned n = 0);


/**
 * Class representing samples series moments estimation using
 */
class Series : public Object {
  public:
    Series(unsigned nMoments, unsigned windowSize);

    ~Series() override = default;

    /**
     * Estimate new k-th moment value from the previous estimation and
     * new samples.
     * @param order moment order (greater or equal then 1)
     * @param value previous estimation
     * @param window array of new samples
     * @param windowSize number of samples to be taken from window
     * @param nRecords total number of samples, incl. those in the window
     * @return new moment estimation
     */
    static double estimate_moment(
            int order,
            double value,
            const std::vector<double> &window,
            unsigned windowSize,
            unsigned nRecords
    );

    /**
     * Record new sample. The sample will be written into the window.
     * If the window is full, then new moments values will be estimated
     * using `commit()`.
     * @param x
     */
    void record(double x);

    /** Estimate new moments values and reset sliding window.
     */
    void commit();

    /** Get estimated moments values. */
    inline const std::vector<double> &moments() const {
        return moments_;
    }

    /** Get moment of the given order. */
    inline double moment(int order) const {
        if (order <= 0 || order > static_cast<int>(moments_.size())) {
            throw std::out_of_range("illegal order");
        }
        return moments_[order - 1];
    }

    /** Get mean value. */
    inline double mean() const { return moments_[0]; }

    /** Get unbiased variance. */
    inline double var() const {
        return getUnbiasedVariance(
                moments_[0],
                moments_[1],
                nCommittedRecords_);
    }

    /** Get standard deviation. */
    inline double std() const { return std::pow(var(), 0.5); }

    /** Get number of recorded samples. */
    inline unsigned count() const { return nRecords_; }

    /** Get string representation of the Series object. */
    std::string toString() const override;

  private:
    std::vector<double> moments_;
    std::vector<double> window_;
    unsigned wPos_;
    unsigned nRecords_;
    unsigned nCommittedRecords_;
};


/**
 * Size distribution given with a probability mass function of
 * values 0, 1, ..., N-1.
 */
class SizeDist : public Object {
  public:
    /**
     * Create size distribution from a given PMF.
     * @param pmf a vector with sum of elements equal 1.0,
     *      all elements should be non-negative.
     */
    SizeDist();
    explicit SizeDist(std::vector<double> pmf);
    SizeDist(const SizeDist &other) = default;
    ~SizeDist() override = default;

    /**
     * Get k-th moment of the distribution.
     * @param order - number of moment (e.g. 1 - mean value)
     * @return sum of i^k * pmf[i] over all i
     */
    double moment(int order) const;

    /** Get mean value. */
    double mean() const;

    /** Get variance. */
    double var() const;

    /** Get standard deviation. */
    double std() const;

    /** Get probability mass function. */
    inline const std::vector<double> &pmf() const {
        return pmf_;
    }

    /** Get string representation. */
    std::string toString() const override;

  private:
    std::vector<double> pmf_;
};


/**
 * Class for recording time-size series, e.g. system or queue size.
 *
 * Size varies in time, so here we store how long each size value
 * was kept. When estimating moments, we just divide all the time
 * on the total time and so value the probability mass function.
 */
class TimeSizeSeries : public Object {
  public:
    explicit TimeSizeSeries(double time = 0.0, unsigned value = 0);

    ~TimeSizeSeries() override;

    /**
     * Record new value update.
     *
     * Here we record information about _previous_ value, and that
     * it was kept for `(time - prevRecordTime)` interval.
     * We also store the new value as `currValue`, so the next
     * time this method is called, information about this value
     * will be recorded.
     *
     * @param time current time
     * @param value new value
     */
    void record(double time, unsigned value);

    /** Estimate probability mass function. */
    std::vector<double> pmf() const;

    /** Get string representation. */
    std::string toString() const override;

  private:
    double initTime_;
    unsigned currValue_;
    double prevRecordTime_;
    std::vector<double> durations_;
};


/**
 * A plain structure-like class representing samples statistics:
 *
 * - average value
 * - standard deviation
 * - variance
 * - number (count) of samples
 * - estimated moments (first N moments)
 *
 * This class doesn't contain any dynamically allocated objects those need
 * manually freeing/deletion.
 */
class VarData : public Object {
  public:
    double mean = 0.0;    ///< Estimated average value
    double std = 0.0;     ///< Estimated standard deviation
    double var = 0.0;     ///< Estimated variance
    unsigned count = 0;   ///< Number of samples used in estimation
    std::vector<double> moments;  ///< First N moments

    VarData() = default;
    VarData(const VarData &other) = default;

    /**
     * Construct VarData from Series object.
     * @param series
     */
    explicit VarData(const Series &series);

    /** Get string representation. */
    std::string toString() const override;
};


/**
 * Simple integer counter that can be evaluated, incremented or reset.
 */
class Counter : public Object {
  public:
    /**
     * Convert from integer constructor.
     * @param initValue initial counter value (default: 0)
     */
    Counter(int initValue = 0); // NOLINT(google-explicit-constructor)

    Counter(const Counter &counter) = default;
    ~Counter() override = default;

    Counter &operator=(const Counter &rside);

    /** Get counter value. */
    inline int value() const { return value_; }

    /** Increment counter value. */
    inline void inc() { value_++; }

    /** Reset counter. */
    inline void reset(int initValue = 0) { value_ = initValue; }

    /** Get string representation. */
    std::string toString() const override;

  private:
    int value_ = 0;
};

}

#endif //CQUMO_TANDEM_STATISTICS_H
