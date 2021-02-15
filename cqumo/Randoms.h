/**
 * @author Andrey Larionov
 */
#ifndef CQUMO_RANDOMS_H
#define CQUMO_RANDOMS_H

#include "Functions.h"
#include <random>
#include <vector>

namespace cqumo {


void *createEngine();
void *createEngineWith(unsigned seed);
void destroyEngine(void *engine);

class RandomVariable;

class Randoms {
  public:
    Randoms();
    Randoms(unsigned seed);
    ~Randoms();

    RandomVariable *createConstant(double value);
    RandomVariable *createExponential(double rate);
    RandomVariable *createUniform(double a, double b);
    RandomVariable *createNormal(double mean, double std);
    RandomVariable *createErlang(int shape, double param);
    
    RandomVariable *createMixture(
      const std::vector<RandomVariable*>& vars,
      const std::vector<double>& weights);
    
    RandomVariable *createHyperExp(
      const std::vector<double>& rates, 
      const std::vector<double>& weights);

    RandomVariable *createAbsorbSemiMarkov(
      const std::vector<RandomVariable*>& vars,
      const std::vector<double>& initProbs,
      const std::vector<std::vector<double>>& transitions,
      int absorbState);
    
    RandomVariable *createChoice(
      const std::vector<double>& values,
      const std::vector<double>& weights);
    
    RandomVariable *createSemiMarkovArrival(
      const std::vector<RandomVariable*>& vars,
      std::vector<double>& initProbs,
      const std::vector<std::vector<double>>& allTransProbs);

  private:
    std::default_random_engine *engine_ = nullptr;
};


class RandomVariable {
  public:
    explicit RandomVariable(void *engine);
    virtual ~RandomVariable() = default;

    inline std::default_random_engine *engine() const { return engine_; }

    virtual double eval() = 0;
  private:
    std::default_random_engine *engine_ = nullptr;
};


class ConstVariable : public RandomVariable {
  public:
    ConstVariable(void *engine, double value);
    ~ConstVariable() override = default;

    double eval() override;
  private:
    double value_ = 0.0;
};


class ExponentialVariable : public RandomVariable {
  public:
    ExponentialVariable(void *engine, double rate);
    ~ExponentialVariable() override = default;

    double eval() override;
  private:
    std::exponential_distribution<double> distribution;
};


class UniformVariable : public RandomVariable {
  public:
    UniformVariable(void *engine, double a, double b);
    ~UniformVariable() override = default;

    double eval() override;
  private:
    std::uniform_real_distribution<double> distribution;
};


class NormalVariable : public RandomVariable  {
  public:
    NormalVariable(void *engine, double mean, double std);
    ~NormalVariable() override = default;

    double eval() override;
  private:
    std::normal_distribution<double> distribution;
};


class ErlangVariable : public RandomVariable {
  public:
    ErlangVariable(void *engine, int shape, double param);
    ~ErlangVariable() override = default;

    double eval() override;
  private:
    int shape_;
    std::exponential_distribution<double> exponent;
};


class HyperExpVariable : public RandomVariable {
  public:
    HyperExpVariable(
      void *engine, 
      const std::vector<double>& rates,
      const std::vector<double>& probs);

    ~HyperExpVariable() override = default;

    double eval() override;
  private:
    std::discrete_distribution<int> choices_;
    std::vector<std::exponential_distribution<double>> exponents_;
};


class MixtureVariable : public RandomVariable {
  public:
    MixtureVariable(
      void *engine,
      const std::vector<RandomVariable*>& vars,
      const std::vector<double> weights);
    
    ~MixtureVariable() override = default;

    double eval() override;
  private:
    std::vector<RandomVariable*> vars_;
    std::vector<double> weights_;
    std::discrete_distribution<int> choices_;
};


class AbsorbSemiMarkovVariable : public RandomVariable {
  public:
    AbsorbSemiMarkovVariable(
      void *engine,
      const std::vector<RandomVariable*>& vars,
      const std::vector<double>& initProbs,
      const std::vector<std::vector<double>>& transitions,
      int absorbState
    );
    ~AbsorbSemiMarkovVariable() override = default;

    double eval() override;
  private:
    std::vector<RandomVariable*> vars_;
    int absorbState_;
    std::discrete_distribution<int> initChoices_;
    std::vector<std::discrete_distribution<int>> transitions_;
};

class ChoiceVariable : public RandomVariable {
  public:
    ChoiceVariable(
      void *engine,
      const std::vector<double>& values,
      const std::vector<double>& weights);
    ~ChoiceVariable() override = default;

    double eval() override;
  private:
    std::vector<double> values_;
    std::discrete_distribution<int> choices_;
};

class SemiMarkovArrivalVariable : public RandomVariable {
  public:
    /**
     * Build SemiMarkovArrivalVariable.
     * 
     * @param engine
     * @param vars random variables for time in each state
     * @param initProbs initial probabilities of size N
     * @param allTransProbs concatenated matrix of shape N x kN, where
     *    k is the number of packet types + 1 (k = 2 for regular MAP).
     *    Looks like [D0 - diag(D0) | D1] / diag(D0).
     */
    SemiMarkovArrivalVariable(
      void *engine,
      const std::vector<RandomVariable*>& vars,
      std::vector<double>& initProbs,
      const std::vector<std::vector<double>>& allTransProbs
    );
    ~SemiMarkovArrivalVariable() override = default;

    double eval() override;
  private:
    std::vector<RandomVariable*> vars_;
    std::discrete_distribution<int> initChoices_;
    std::vector<std::discrete_distribution<int>> allTransChoices_;
    int state_;
    int order_;
};

}

#endif //CQUMO_RANDOMS_H
