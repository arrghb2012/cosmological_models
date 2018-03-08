#ifndef LAMBDACDM_HPP
#define LAMBDACDM_HPP

#include "CosmologicalModel.hpp"

class LambdaCDM_model: public CosmologicalModel
{
public:
    virtual ~LambdaCDM_model() {};

public:
    double atoH(double a) override;

};

#endif
