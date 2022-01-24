#include <iostream>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <random>
#include <cmath>
#include "LHAPDF/LHAPDF.h"
#include <gsl/gsl_math.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_vegas.h>

using namespace LHAPDF;
using namespace std;

struct params {
	double m_Z = 91.1876;
	double m_W = 80.403;
	double pi = 2 * acos(0.0);
	double G_F = 1.16637e-5;
	double sin_thetaW = 1 - pow(m_W,2) / pow(m_Z, 2);
	double alpha = sqrt(2) * G_F * pow(m_W, 2) * sin_thetaW / pi;
	double e = sqrt(4 * alpha * pi);
	double e_l[2];
	double thetaf = 82.0/360*2*pi;
	double m_f; //= 114.8;
	double S = pow(14e3, 2);
	double T_f[2], e_q[2], L[2], R[2], lL[2], lR[2];
	double S_ij[2][2], delta[2][2], sL[2][2][2], sR[2][2][2];
	const PDF* pdf;
	int pid;
	double M2;
	int lepton_type;
};

double test_LOcross(double M2, int a, int L_ID, void* p) {
	struct params* fp = (struct params*)p;

	double LO_xsec = 0;

	double beta = sqrt(1 + pow(fp->m_f, 4) / pow(M2, 2) + pow(fp->m_f, 4) / pow(M2, 2) - 2 * (pow(fp->m_f, 2) / M2 + pow(fp->m_f, 2) / M2 + (pow(fp->m_f, 2) * pow(fp->m_f, 2)) / pow(M2, 2)));
	double frac = pow(fp->alpha, 2) * fp->pi * pow(beta, 3) / (9 * M2);

	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < 1; j++) {
			double term1 = pow(fp->e_q[a], 2) * pow(fp->e_l[L_ID], 2) * fp->delta[i][j];
			double term2 = fp->e_q[a] * fp->e_l[L_ID] * fp->delta[i][j] * (fp->L[a] + fp->R[a]) * (fp->sL[L_ID][i][j] + fp->sR[L_ID][i][j]) / (4 * fp->sin_thetaW * (1 - fp->sin_thetaW) * (1 - pow(fp->m_Z, 2) / M2));
			double term3 = (pow(fp->L[a], 2) + pow(fp->R[a], 2)) * pow((fp->sL[L_ID][i][j] + fp->sR[L_ID][i][j]), 2) / (32 * pow(fp->sin_thetaW, 2) * pow((1 - fp->sin_thetaW), 2) * pow((1 - pow(fp->m_Z, 2) / M2), 2));

			LO_xsec += frac * (term1 + term2 + term3);
		}
	}
	return LO_xsec;
}

double LO_integrand(double x[], size_t dim, void* p) {
	struct params* fp = (struct params*)p;
	int a = abs((fp->pid)) % 2;
	
	int L_ID = fp->lepton_type;

	if (x[0] * x[1] * fp->S < 4 * pow(fp->m_f, 2)) {
		return 0;
	}
	else {
		double M2 = x[0]*x[1]*fp->S;

		double partonic_xsec = test_LOcross(M2, a, L_ID, fp);

		/*double partonic_xsec = 0;

		double beta = sqrt(1 + pow(fp->m_f, 4) / pow(M2, 2) + pow(fp->m_f, 4) / pow(M2, 2) - 2 * (pow(fp->m_f, 2) / M2 + pow(fp->m_f, 2) / M2 + (pow(fp->m_f, 2) * pow(fp->m_f, 2)) / pow(M2, 2)));
		double frac = pow(fp->alpha, 2) * fp->pi * pow(beta, 3) / (9 * M2);

		for (int i = 0; i < 1; i++) {
			for (int j = 0; j < 1; j++) {
				double term1 = pow(fp->e_q[a], 2) * pow(fp->e_l[L_ID], 2) * fp->delta[i][j];
				double term2 = fp->e_q[a] * fp->e_l[L_ID] *fp->delta[i][j] * (fp->L[a] + fp->R[a])* (fp->sL[L_ID][i][j] + fp->sR[L_ID][i][j]) / (4 * fp->sin_thetaW * (1 - fp->sin_thetaW) * (1 - pow(fp->m_Z, 2) / M2));
				double term3 = (pow(fp->L[a], 2) + pow(fp->R[a], 2)) * pow((fp->sL[L_ID][i][j] + fp->sR[L_ID][i][j]), 2) / (32 * pow(fp->sin_thetaW, 2) * pow((1 - fp->sin_thetaW), 2) * pow((1 - pow(fp->m_Z, 2) / M2), 2));

				partonic_xsec += frac * (term1 + term2 + term3);
			}
		}*/
		return (fp->pdf->xfxQ2(fp->pid, x[0], pow(fp->m_f, 2)) / x[0] * fp->pdf->xfxQ2(-1*(fp->pid), x[1], pow(fp->m_f, 2)) / x[1] * partonic_xsec);
	}
	
}

void display_results(char const* title, double result, double error)
{
	printf("%s ==================\n", title);
	printf("result = % .6e\n", result);
	printf("sigma  = % .6e\n", error);
	//printf("exact  = % .6f\n", exact);
	//printf("error  = % .6e = %.2g sigma\n", result - exact,
		//fabs(result - exact) / error);
}

int main(int argc, char* argv[]) {

	struct params p;
	//for quarks: 1 is down-type and 0 is up-type

	p.e_q[1] = -1.0 * (1.0 / 3); p.e_q[0] = (2.0 / 3);
	p.e_l[1] = -1; p.e_l[0] = 0;
	p.S_ij[0][0] = cos(p.thetaf); p.S_ij[0][1] = sin(p.thetaf);
	p.S_ij[1][0] = -sin(p.thetaf); p.S_ij[1][1] = cos(p.thetaf);

	for (int i = 0; i < 2; i++) {
		p.T_f[i] = pow(-1, i) * (1.0 / 2);
		p.L[i] = 2*p.T_f[i] - 2*p.e_q[i] * p.sin_thetaW;
		p.R[i] = -2*p.e_q[i] * p.sin_thetaW;
		p.lL[i] = 2*p.T_f[i] - 2 *p.e_l[i]* p.sin_thetaW;
		p.lR[i] = -2 *p.e_l[i]*p.sin_thetaW;

		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {
				p.sL[i][j][k] = p.lL[i] * p.S_ij[j][0] * p.S_ij[k][0];
				p.sR[i][j][k] = p.lR[i] * p.S_ij[j][1] * p.S_ij[k][1];
			}
			if (i == j) {
				p.delta[j][i] = 1;
			}
			else {
				p.delta[j][i] = 0;
			}
		}
	}


	printf("%.5f\n", p.thetaf);
	if (argc < 3) {
		cerr << "You must specify a PDF set and member number" << endl;
		return 1;
	}

	const string setname = argv[1];
	const string smem = argv[2];
	const int imem = lexical_cast<int>(smem);
	p.pdf = mkPDF(setname, imem);

	printf("going through params: %.8f & %.8f, %.8f, %.8f\n", p.T_f[1], p.sin_thetaW, p.S, 1/p.alpha);

	vector<int> pids = p.pdf->flavors();

	for (int i = 0; i < 10; i++) {
		double min_mf = 114.8;
		double max_mf = 350;
		double dm = (max_mf - min_mf) / 10;
		p.m_f = min_mf + i * dm;
		p.lepton_type = 1;

		double xsec = 0;

		//double tau = p.M2 / p.S;
		double tau = 4 * pow(p.m_f, 2) / p.S;
		//printf("%.4f\n", p.thetaf);

		size_t dim = 2;

		double x0[dim] = { tau, tau };
		double x1[dim] = { 1.0, 1.0 };

		const gsl_rng_type* T;
		gsl_rng* r;

		for (int a = 0; a < 10; a++) {
			p.pid = pids[a];
			//printf("%d\n", p.pid);

			gsl_monte_function G = { &LO_integrand, dim, &p };

			size_t calls = 100000;

			gsl_rng_env_setup();

			T = gsl_rng_default;
			r = gsl_rng_alloc(T);

			double res = 0;
			double err = 0;
			{
				gsl_monte_vegas_state* s = gsl_monte_vegas_alloc(dim);
				//printf("starting vegas: \n");
				gsl_monte_vegas_integrate(&G, x0, x1, dim, 10000, r, s, &res, &err);

				//display_results("vegas warm-up", res, err);

				//printf("converging...\n");

				do
				{
					gsl_monte_vegas_integrate(&G, x0, x1, dim, calls / 5, r, s,
						&res, &err);
					//printf("result = % .6e sigma = % .6e chisq/dof = %.3e\n", res, err, gsl_monte_vegas_chisq(s));
				} while (fabs(gsl_monte_vegas_chisq(s) - 1) > 0.5);

				//display_results("vegas final", res, err);

				gsl_monte_vegas_free(s);
			}

			xsec += res;
		}

		gsl_rng_free(r);

		xsec *= 0.38938e-3;
		printf("Final res after summing up Parton ID's: %.5e with stau mass: %.2f\n", xsec, p.m_f);
	}

	

	delete p.pdf;

	return 0;
}