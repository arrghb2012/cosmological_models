#ifndef SIGMACDM_HPP
#define SIGMACDM_HPP

#include "CosmologicalModel.hpp"

class SigmaCDM_model: public CosmologicalModel
{
public:
    virtual ~SigmaCDM_model() {};
    double atoH(double a) override;
    void setOmega_curv0() override;
    void set_tildeA(double tildeA);
    void set_tildeB(double tildeB);
    void set_nu(double nu);
    double get_tildeA();
    double get_tildeB();
    double get_nu();
    
private:
    double mtildeA, mtildeB, mnu;

};

#endif
