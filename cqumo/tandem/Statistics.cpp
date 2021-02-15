/**
 * @author Andrey Larionov
 */
#include "Statistics.h"
#include <cmath>
#include <sstream>
#include <ostream>
#include <iterator>


namespace cqumo {

double getUnbiasedVariance(double m1, double m2, unsigned n) {
    if (n > 1) {
        auto _n = static_cast<double>(n);
        return (m2 - m1 * m1) * (_n / (_n - 1));
    }
    return m2 - m1 * m1;
}


// Class Series
// --------------------------------------------------------------------------

Series::Series(unsigned nMoments, unsigned windowSize) {
    moments_.resize(nMoments, 0.0);
    window_.resize(windowSize, 0.0);
    wPos_ = 0;
    nRecords_ = 0;
    nCommittedRecords_ = 0;
}

void Series::record(double x) {
    window_[wPos_++] = x;
    nRecords_++;
    if (wPos_ >= window_.size()) {
        commit();
    }
}

void Series::commit() {
    int numMoments = static_cast<int>(moments_.size());
    for (int i = 0; i < numMoments; ++i) {
        moments_[i] = estimate_moment(
                i + 1,
                moments_[i],
                window_, wPos_, nRecords_);
    }
    nCommittedRecords_ = nRecords_;
    wPos_ = 0;
}

std::string Series::toString() const {
    std::stringstream ss;
    ss << "(Series: moments=[";
    std::copy(moments_.begin(), moments_.end(),
              std::ostream_iterator<double>(ss, " "));
    ss << "], nRecords=" << nRecords_ << ")";
    return ss.str();
}

double Series::estimate_moment(
        int order,
        double value,
        const std::vector<double> &window,
        unsigned windowSize,
        unsigned nRecords) {
    if (nRecords <= 0) {
        return value;
    }
    double accum = 0.0;
    windowSize = std::min(static_cast<unsigned>(window.size()), windowSize);
    for (unsigned i = 0; i < windowSize; ++i) {
        accum += std::pow(window[i], order);
    }
    return value * (1.0 - static_cast<double>(windowSize) / nRecords) +
           accum / nRecords;
}


// Class SizeDist
// --------------------------------------------------------------------------

SizeDist::SizeDist() : pmf_(std::vector<double>(1, 1.0)) {}

SizeDist::SizeDist(std::vector<double> pmf) : pmf_(std::move(pmf)) {}

double SizeDist::moment(int order) const {
    double accum = 0.0;
    for (unsigned i = 0; i < pmf_.size(); ++i) {
        accum += std::pow(i, order) * pmf_[i];
    }
    return accum;
}

double SizeDist::mean() const {
    return moment(1);
}

double SizeDist::var() const {
    return moment(2) - std::pow(moment(1), 2);
}

double SizeDist::std() const {
    return std::pow(var(), 0.5);
}

std::string SizeDist::toString() const {
    std::stringstream ss;
    ss << "(SizeDist: mean=" << mean() << ", std=" << std()
       << ", pmf=" << cqumo::toString(pmf_) << ")";
    return ss.str();
}


// Class TimeSizeSeries
// --------------------------------------------------------------------------

TimeSizeSeries::TimeSizeSeries(double time, unsigned value)
        : initTime_(time), currValue_(value), prevRecordTime_(0.0) {
    durations_.resize(1, 0.0);
}

TimeSizeSeries::~TimeSizeSeries() = default;

void TimeSizeSeries::record(double time, unsigned value) {
    if (durations_.size() <= currValue_) {
        durations_.resize(currValue_ + 1, 0.0);
    }
    durations_[currValue_] += time - prevRecordTime_;
    prevRecordTime_ = time;
    currValue_ = value;
}

std::vector<double> TimeSizeSeries::pmf() const {
    std::vector<double> pmf(durations_);
    double dt = prevRecordTime_ - initTime_;
    for (double & i : pmf) {
        i /= dt;
    }
    return pmf;
}

std::string TimeSizeSeries::toString() const {
    std::stringstream ss;
    ss << "(TimeSizeSeries: durations=[";
    std::copy(durations_.begin(), durations_.end(),
              std::ostream_iterator<double>(ss, " "));
    ss << "])";
    return ss.str();
}


// Class VarData
// --------------------------------------------------------------------------

VarData::VarData(const Series &series)
        : mean(series.mean()),
          std(series.std()),
          var(series.var()),
          count(series.count()),
          moments(series.moments()) {}

std::string VarData::toString() const {
    std::stringstream ss;
    ss << "(VarData: mean=" << mean
       << ", var=" << var
       << ", std=" << std
       << ", count=" << count
       << ", moments=[" << cqumo::toString(moments) << "])";
    return ss.str();
}


// Class Counter
// --------------------------------------------------------------------------

Counter::Counter(int initValue) : value_(initValue) {}

Counter &Counter::operator=(const Counter &rside) {
    value_ = rside.value();
    return *this;
}

std::string Counter::toString() const {
    std::stringstream ss;
    ss << "(Counter: value=" << value_ << ")";
    return ss.str();
}


}
