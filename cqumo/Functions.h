/**
 * Module with function type definitions.
 * Provides ContextFunctor class that can be used to bind intervals generating
 * functions from Python code.
 *
 * @author Andrey Larionov
 */
#ifndef CQUMO_FUNCTIONS_H
#define CQUMO_FUNCTIONS_H

#include <functional>

namespace cqumo {

typedef std::function<double()> DblFn;
typedef std::function<double(void *)> CtxDblFn;


/**
 * Class that can be used as DblFn, while using a context for calling
 * a context-dependent function of type CtxDblFn.
 * Typically this functor can be used to bind Python functions (e.g.
 * Distribution or RandomProcess instances) into simulations where
 * DblFn is expected.
 */
class ContextFunctor {
  public:
    /**
     * Create a functor.
     * @param fn function with spec `(void*) -> double`
     * @param context argument that is passed to fn on each call
     */
    explicit ContextFunctor(const CtxDblFn &fn, void *context);
    ContextFunctor(const ContextFunctor &other) = default;

    ContextFunctor &operator=(const ContextFunctor &other) = default;

    /** Makes a call to fn with context as argument. */
    inline double operator()() const {
        return fn_(context_);
    }
  private:
    CtxDblFn fn_;
    void *context_;
};


DblFn makeDblFn(CtxDblFn ctxFn, void *context);

}

#endif //FUNCTIONS_H
