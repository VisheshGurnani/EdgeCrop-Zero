#pragma once
#include <cstdarg>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class DecisionTree {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        if (x[0] <= 34.978994369506836) {
                            return 1;
                        }

                        else {
                            if (x[1] <= 59.86634635925293) {
                                if (x[2] <= 34.673038482666016) {
                                    if (x[2] <= 31.40719699859619) {
                                        return 1;
                                    }

                                    else {
                                        return 1;
                                    }
                                }

                                else {
                                    if (x[3] <= 89.84074020385742) {
                                        return 1;
                                    }

                                    else {
                                        return 0;
                                    }
                                }
                            }

                            else {
                                return 0;
                            }
                        }
                    }

                protected:
                };
            }
        }
    }