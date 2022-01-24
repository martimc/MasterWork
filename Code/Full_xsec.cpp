#include <iostream>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <random>
#include <cmath>
#include <complex> 
#include "LHAPDF/LHAPDF.h"
#include <gsl/gsl_math.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_vegas.h>
#include "clooptools.h"

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
	double m_g = 2e3; 
	double S = pow(14e3, 2);
	double T_f[2], e_q[2], L[2], R[2], lL[2], lR[2], m_sq[2];
	double S_ij[2][2], delta[2][2], sL[2][2][2], sR[2][2][2];
	const PDF* pdf;
	int pid;
	double M2;
	int lepton_type;
	double mu_F;
	double mu_R;
	double alpha_s;
};

double f_gamma(double M2, double m_gluino, double m_sq[2]) {
	double result = 2;
	ltini();
	for (int k = 0; k < 2; k++) {
		double mq2 = pow(m_sq[k], 2);
		double term1 = (2 * m_gluino - 2 * m_sq[k] + M2) / M2 * (real(B0(M2, mq2, mq2)) - real(B0(0, mq2, mq2)));
		double term2 = (mq2 - pow(m_gluino, 2)) * real(DB0(0, pow(m_gluino, 2), mq2));
		double term3 = 2 * (pow(m_gluino, 4) + (M2 - 2 * mq2) * pow(m_gluino, 2) + pow(mq2, 2)) / M2 * real(C0(0, M2, 0, mq2, pow(m_gluino, 2), mq2));

		result += term1 + term2 + term3;
	}
	//ltexi();
	return result;
}

double susy_cross(double M2, int a, int L_ID, void* p) {
	struct params* fp = (struct params*)p;

	double susy_xsec = 0;
	double C_F = 4.0 / 3;

	double beta = sqrt(1 + pow(fp->m_f, 4) / pow(M2, 2) + pow(fp->m_f, 4) / pow(M2, 2) - 2 * (pow(fp->m_f, 2) / M2 + pow(fp->m_f, 2) / M2 + (pow(fp->m_f, 2) * pow(fp->m_f, 2)) / pow(M2, 2)));
	double frac = pow(fp->alpha, 2) * fp->pi * C_F * pow(beta, 3) / (36 * M2);

	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < 1; j++) {
			double term1 = pow(fp->e_q[a], 2) * pow(fp->e_l[L_ID], 2) * fp->delta[i][j];//f_gamma(M2, fp->m_g, fp->m_sq)*pow(fp->e_q[a], 2) * pow(fp->e_l[L_ID], 2) * fp->delta[i][j];
			double term2 = fp->e_q[a] * fp->e_l[L_ID] * fp->delta[i][j] * (fp->L[a] + fp->R[a]) * (fp->sL[L_ID][i][j] + fp->sR[L_ID][i][j]) / (4 * fp->sin_thetaW * (1 - fp->sin_thetaW) * (1 - pow(fp->m_Z, 2) / M2));
			double term3 = (pow(fp->L[a], 2) + pow(fp->R[a], 2)) * pow((fp->sL[L_ID][i][j] + fp->sR[L_ID][i][j]), 2) / (32 * pow(fp->sin_thetaW, 2) * pow((1 - fp->sin_thetaW), 2) * pow((1 - pow(fp->m_Z, 2) / M2), 2));

			susy_xsec += frac * (term1 + term2 + term3);
		}
	}
	return susy_xsec;
}

double susy_integrand(double x[], size_t dim, void* p) {
	struct params* fp = (struct params*)p;

	double M2 = x[0] * x[1] * fp->S;

	if (M2 < 4 * pow(fp->m_f, 2)) {
		return 0;
	}
	else {
		int a = abs((fp->pid)) % 2;
		int L_ID = fp->lepton_type;

		double partonic_xsec = susy_cross(M2, a, L_ID, fp);

		return (fp->pdf->xfxQ2(fp->pid, x[0], pow(fp->mu_F, 2)) / x[0] * fp->pdf->xfxQ2(-1 * (fp->pid), x[1], pow(fp->mu_F, 2)) / x[1] * partonic_xsec);
	}
}

double LO_cross(double M2, int a, int L_ID, void* p) {
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

	double M2 = x[0] * x[1] * fp->S;

	if (M2 < 4 * pow(fp->m_f, 2)) {
		return 0;
	}
	else {
		int a = abs((fp->pid)) % 2;
		int L_ID = fp->lepton_type;

		double partonic_xsec = LO_cross(M2, a, L_ID, fp);

		return (fp->pdf->xfxQ2(fp->pid, x[0], pow(fp->mu_F, 2)) / x[0] * fp->pdf->xfxQ2(-1 * (fp->pid), x[1], pow(fp->mu_F, 2)) / x[1] * partonic_xsec);
	}
}

double P_qg(double z, double T_R) {
	return T_R / 2 * (pow(z, 2) + pow((1 - z), 2));
}

double Z_NLO_integrand(double x[], size_t dim, void* p) {
	struct params* fp = (struct params*)p;

	double M2 = x[0] * x[1] * x[2] * fp->S;
	double z = x[0];

	if (M2 < 4 * pow(fp->m_f, 2)) {
		return 0;
	}
	else {
		int a = abs((fp->pid)) % 2;
		int L_ID = fp->lepton_type;
		double C_F = 4.0 / 3;
		double T_R = 1.0 / 2;

		double LO_xsec = LO_cross(M2, a, L_ID, fp);
		double frac = fp->alpha_s / (fp->pi) * LO_xsec * (fp->pdf->xfxQ2(fp->pid, x[0], pow(fp->m_f, 2)) / x[0]);

		double z2 = pow(z, 2);

		double term1 = -1 * (1 + z2) / (1 - z) * log(z);
		double term2 = -2 * (1 + z) * log(1 - z);
		double term3 = -1 * (1 + z) * log(M2 / pow(fp->mu_F, 2));

		double qq_term = fp->pdf->xfxQ2(-1 * (fp->pid), x[1], pow(fp->m_f, 2)) / x[1] * C_F * (term1 + term2 + term3);

		double qg_terms = (1.0 / 2 - z + z2) * log(pow((1 - z), 2) / z) + 1.0 / 4 + 3 * z / 2 - 7 * z2 / 4 + P_qg(z, T_R) / T_R * log(M2 / pow(fp->mu_F, 2));

		double qg_term = fp->pdf->xfxQ2(21, x[1], pow(fp->m_f, 2)) / x[1] * T_R * qg_terms;

		double partonic_xsec = frac * (qq_term + qg_term);

		return partonic_xsec;
	}
}

double xa_xb_integrand(double x[], size_t dim, void* p) {
	struct params* fp = (struct params*)p;
	
	double M2 = x[0] * x[1] * fp->S;

	if (M2 < 4 * pow(fp->m_f, 2)) {
		return 0;
	}
	else {
		int a = abs((fp->pid)) % 2;
		int L_ID = fp->lepton_type;
		double C_F = 4.0 / 3;

		double LO_xsec = LO_cross(M2, a, L_ID, fp);

		double NLO_terms = pow(fp->pi, 2) / 3 - 4 + 3.0 / 2 * log(M2 / pow(fp->mu_F, 2));

		double partonic_xsec = LO_xsec * (1 + fp->pdf->alphasQ2(pow(fp->mu_R, 2)) / (fp->pi) * C_F * NLO_terms);

		return (fp->pdf->xfxQ2(fp->pid, x[0], pow(fp->mu_F, 2)) / x[0] * fp->pdf->xfxQ2(-1 * (fp->pid), x[1], pow(fp->mu_F, 2)) / x[1] * partonic_xsec);
	}
}

double f_plus(double z) {
	return log(1 - z) / (1 - z);
}

double plus_Integrand(double x[], size_t dim, void* p) {
	struct params* fp = (struct params*)p;

	double M2 = x[0] * x[1] * x[2] * fp->S;

	if (M2 < 4 * pow(fp->m_f, 2)) {
		return 0;
	}
	else {
		int a = abs((fp->pid)) % 2;
		int L_ID = fp->lepton_type;
		double z = x[0];
		double C_F = 4.0 / 3;

		double LO_xsec = LO_cross(M2, a, L_ID, fp);
		double M2_Z1 = x[1] * x[2] * fp->S;
		double LO_xsecZ1 = LO_cross(M2_Z1, a, L_ID, fp);

		double partonic_xsec = 4 * C_F * (f_plus(z) * (LO_xsec - LO_xsecZ1) + LO_xsecZ1 * log(1 - z));

		return (fp->pdf->xfxQ2(fp->pid, x[1], pow(fp->mu_F, 2)) / x[1] * fp->pdf->xfxQ2(-1 * (fp->pid), x[2], pow(fp->mu_F, 2)) / x[2] * partonic_xsec);
	}
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
		p.m_sq[i] = 1e3;

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

	for (int i = 0; i < 11; i++) {
		double min_mf = 114.8;
		double max_mf = 350;
		double dm = (max_mf - min_mf) / 10;
		p.m_f = min_mf + i * dm;
		p.mu_F = p.m_f;
		p.mu_R = p.m_f;
		p.alpha_s = p.pdf->alphasQ2(pow(p.mu_R, 2));
		p.lepton_type = 1;

		double xsec = 0;
		double xsec_LO = 0;

		//double tau = p.M2 / p.S;
		double tau = 4 * pow(p.m_f, 2) / p.S;
		//printf("%.4f\n", p.thetaf);

		size_t dim1 = 2;
		size_t dim2 = 3;

		double x0[dim1] = { tau, tau };
		double x1[dim1] = { 1.0, 1.0 };

		double x0_NLO[dim2] = { tau, tau, tau };
		double x1_NLO[dim2] = { 1.0, 1.0, 1.0 };

		//double x0_plus[dim2] = { 0, tau, tau };
		//double x1_plus[dim2] = { 1.0, 1.0, 1.0 };

		const gsl_rng_type* T;
		gsl_rng* r;

		for (int a = 0; a < 10; a++) {
			p.pid = pids[a];

			gsl_monte_function G = { &xa_xb_integrand, dim1, &p };
			gsl_monte_function H = { &Z_NLO_integrand, dim2, &p };
			gsl_monte_function plus = { &plus_Integrand, dim2, &p };
			gsl_monte_function LO = { &susy_integrand, dim1, &p };

			size_t calls = 100000;

			gsl_rng_env_setup();

			T = gsl_rng_default;
			r = gsl_rng_alloc(T);

			double res1 = 0;
			double err1 = 0;

			double res2 = 0;
			double err2 = 0;

			double res_LO = 0;
			double err_LO = 0;

			{
				gsl_monte_vegas_state* s = gsl_monte_vegas_alloc(dim1);
				gsl_monte_vegas_state* t = gsl_monte_vegas_alloc(dim2);
				gsl_monte_vegas_state* u = gsl_monte_vegas_alloc(dim1);

				gsl_monte_vegas_integrate(&G, x0, x1, dim1, 10000, r, s, &res1, &err1);

				do
				{
					gsl_monte_vegas_integrate(&G, x0, x1, dim1, calls / 5, r, s,
						&res1, &err1);
					//printf("result = % .6e sigma = % .6e chisq/dof = %.3e\n", res, err, gsl_monte_vegas_chisq(s));
				} while (fabs(gsl_monte_vegas_chisq(s) - 1) > 0.5);

				gsl_monte_vegas_free(s);
		
				gsl_monte_vegas_integrate(&H, x0_NLO, x1_NLO, dim2, 10000, r, t, &res2, &err2);

				do
				{
					gsl_monte_vegas_integrate(&H, x0_NLO, x1_NLO, dim2, calls / 5, r, t,
						&res2, &err2);
					//printf("result = % .6e sigma = % .6e chisq/dof = %.3e\n", res, err, gsl_monte_vegas_chisq(s));
				} while (fabs(gsl_monte_vegas_chisq(t) - 1) > 0.5);

				gsl_monte_vegas_free(t);

				gsl_monte_vegas_integrate(&LO, x0, x1, dim1, 10000, r, u, &res_LO, &err_LO);

				do
				{
					gsl_monte_vegas_integrate(&LO, x0, x1, dim1, calls / 5, r, u,
						&res_LO, &err_LO);
					printf("result = % .6e sigma = % .6e chisq/dof = %.3e, while term: %.3e\n", res_LO, err_LO, gsl_monte_vegas_chisq(u), fabs(gsl_monte_vegas_chisq(u) - 1));
				} while (fabs(gsl_monte_vegas_chisq(u) - 1) > 0.5);

				gsl_monte_vegas_free(u);
				printf("done with pID: %d\n", p.pid);
			}

			xsec += res1 + res2;// +res_plus;
			xsec_LO += res_LO;
		}

		gsl_rng_free(r);

		xsec *= 0.38938e-3;
		xsec_LO *= 0.38938e-3;
		printf("Final NLO res after summing up Parton ID's: %.5e with stau mass: %.2f\n", xsec, p.m_f);
		printf("Final SLO res after summing up Parton ID's: %.5e with stau mass: %.2f\n", xsec_LO, p.m_f);
	}

	

	delete p.pdf;

	return 0;
}