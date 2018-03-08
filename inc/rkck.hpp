#ifndef _RKCK_HPP_
#define _RKCK_HPP_

#include <vector>
#include <cmath>
#include <algorithm>

struct RKCK{
    double m_x;
    std::vector<double> m_y;
    double m_yStop, m_h;
    double m_tol;
    std::vector<std::vector<double> > K_div_by_h_2darray;
    std::vector<double> C, D;
    std::vector<double> E;
    double e;
    double hNext;
    std::vector<double> yt;
    std::vector<double> yout, dy;
    size_t size_of_y;

    RKCK(const double x, const std::vector<double>& y, const double yStop, const double h = 1.0e-6, const double tol = 1.0e-6):
	m_x(x), m_y(y), m_yStop(yStop), m_h(h), m_tol(tol) {};

    template <class Problem>
    std::vector<double> run_kut5(const Problem& P, const double x, const std::vector<double>& y, const double h){
	P.rhs(x, y, K_div_by_h_2darray[0]);
	for (size_t i = 0; i < size_of_y; ++i)
	{
	    yt[i] = y[i] + h * 1./5 * K_div_by_h_2darray[0][i];  
	}
	P.rhs(x + 1./5*h, yt, K_div_by_h_2darray[1]);
	for (size_t i = 0; i < size_of_y; ++i)
	{
	    yt[i] = y[i] + 3./40 * h * K_div_by_h_2darray[0][i] + 9./40 * h * K_div_by_h_2darray[1][i];  
	}
	P.rhs(x + 3./10 * h, yt, K_div_by_h_2darray[2]);
	for (size_t i = 0; i < size_of_y; ++i)
	{
	    yt[i] = y[i] + 3./10 * h * K_div_by_h_2darray[0][i] 
		- 9./10 * h * K_div_by_h_2darray[1][i] + 6./5 * h * K_div_by_h_2darray[2][i];  
	}
	P.rhs(x + 3./5 * h, yt, K_div_by_h_2darray[3]);
	for (size_t i = 0; i < size_of_y; ++i)
	{
	    yt[i] = y[i] - 11./54 * h * K_div_by_h_2darray[0][i] + 5./2 * h * K_div_by_h_2darray[1][i] 
		- 70./27 * h * K_div_by_h_2darray[2][i] + 35./27 * h * K_div_by_h_2darray[3][i];  
	}
	P.rhs(x + h, yt, K_div_by_h_2darray[4]);
	for (size_t i = 0; i < size_of_y; ++i)
	{
	    yt[i] = y[i] + 1631./55296 * h * K_div_by_h_2darray[0][i] + 175./512 * h * K_div_by_h_2darray[1][i] 
		+ 575./13824 * h * K_div_by_h_2darray[2][i] + 44275./110592 * h * K_div_by_h_2darray[3][i] 
		+ 253./4096 * h * K_div_by_h_2darray[4][i];  
	}
	P.rhs(x + 7./ 8 * h, yt, K_div_by_h_2darray[5]);
	for (size_t i = 0; i < size_of_y; ++i)
	{
	    yout[i] = 0.0;
	    E[i] = 0;		// нужно обнулять значения членов класса перед каждым новым шагом run_kut5
	    for (int j = 0; j < 6; ++j)
	    {
		yout[i] += h * C[j] * K_div_by_h_2darray[j][i];
		E[i] += h * (C[j] - D[j]) * K_div_by_h_2darray[j][i];
	    }
	}
	this->e = 0.0;		// нужно обнулять значения членов класса перед каждым новым шагом run_kut5
	for (size_t i = 0; i < size_of_y; ++i)
	{
	    this->e += (E[i] * E[i] / size_of_y);
	}
	this->e = sqrt(this->e);
	return yout;
    }

    template <class Problem>
    void integrate(const Problem& P, std::vector<double>& X, std::vector<std::vector<double>>& Y){
	size_of_y = m_y.size();
	C = {37./ 378, 0., 250./ 621, 125./594, 0., 512./1771};
	D = {2825./ 27648, 0., 18575./ 48384, 13525./55296, 277./14336, 1./4};
	std::vector<double> ki_div_by_h(size_of_y);
	for (int i = 0; i < 6; ++i)
	{
	    K_div_by_h_2darray.push_back(ki_div_by_h);
	}
	E.resize(size_of_y);
	yt.resize(size_of_y);
	yout.resize(size_of_y);
	dy.resize(size_of_y);
    
	X.push_back(m_x);
	Y.push_back(m_y);

	int stopper = 0;
	int n_steps = 10000;
	for (int i = 0; i < n_steps; ++i)
	{
	    dy = run_kut5(P, m_x, m_y, m_h);
	    if (this->e <= this->m_tol)
	    {
		m_x = m_x + m_h;
		for (size_t j = 0; j < size_of_y; ++j)
		{
		    m_y[j] += dy[j];
		}
		X.push_back(m_x);
		Y.push_back(m_y);
		if (stopper == 1) break;
	    }
	    if (this->e != 0.0)
	    {
		this->hNext = 0.9 * m_h * pow(m_tol / this->e, 0.2);
	    }
	    else hNext = m_h;
	    if ((m_h > 0.0) == (m_y[0] >= m_yStop))
	    {
		stopper = 1;
	    }
	    m_h = hNext;
	}
    }
};


#endif /* _RKCK_HPP_ */

