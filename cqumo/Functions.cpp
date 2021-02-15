/**
 * @author Andrey Larionov
 */
#include "Functions.h"

namespace cqumo {

// Class ContextFunctor
// --------------------------------------------------------------------------
ContextFunctor::ContextFunctor(const CtxDblFn &fn, void *context)
: fn_(fn), context_(context) {}

// Utilities
// --------------------------------------------------------------------------
DblFn makeDblFn(CtxDblFn ctxFn, void *context) {
    return ContextFunctor(ctxFn, context);
}

}
