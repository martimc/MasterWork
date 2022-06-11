#include <iostream>
#include <typeinfo>
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
//#include "clooptools.h"

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
	double thetaf = 0; //6.25e-01//82.0/360*2*pi;
	double m_f; //= 114.8;
	double m_g = 2e3;
	double S = pow(13e3, 2);
	double T_f[2], e_q[2], L[2], R[2], lL[2], lR[2], m_sq[2];
	double S_ij[2][2], delta[2][2], sL[2][2][2], sR[2][2][2], sLq[2][2][2], sRq[2][2][2];
	vector<LHAPDF::PDF*> pdfs;
	int pid;
	double M2;
	int lepton_type;
	double mu_F;
	double mu_R;
	double alpha_s;
	vector<int> pids;
	size_t imems;
};

/*double f_gamma(double M2, double m_gluino, double m_sq[2]) {
	double result = 2;
	for (int k = 0; k < 2; k++) {
		double mq2 = pow(m_sq[k], 2);
		double mg2 = pow(m_gluino, 2);
		double term1 = (2 * mg2 - 2 * mq2 + M2) / M2 *(real(B0(M2, mq2, mq2)) - real(B0(0, mg2, mq2)));
		double term2 = (mq2 - mg2)*real(DB0(0, mg2, mq2));
		double term3 = 2 * (pow(mg2, 2) + (M2 - 2 * mq2) * mg2 + pow(mq2, 2)) / M2 * real(C0(0, M2, 0, mq2, mg2, mq2));

		result += term1 + term2 + term3;
	}
	//ltexi();
	return result;
}

double f_gammaZ(double M2, int a, void* p) {
	struct params* fp = (struct params*)p;
	double L_qq = fp->L[a];
	double R_qq = fp->R[a];
	double result = 2 * (L_qq + R_qq);

	for (int i = 0; i < 2; i++) {
		double mg2 = pow(fp->m_g, 2);
		double mq2 = pow(fp->m_sq[i], 2);
		double sL_qq = fp->sLq[a][i][i];
		double sR_qq = fp->sRq[a][i][i];
		double S_i = pow(fp->S_ij[i][0],2);

		double term1 = 2 * (2 * mg2 - 2 * mq2 + M2) * (sL_qq + sR_qq) / M2 * real(B0(M2, mq2, mq2));
		double term2 = 2 * (2 * mg2 - 2 * mq2 + M2) * (L_qq * S_i + R_qq * S_i) / M2 * real(B0(0, mg2, mq2));
		double term3 = (mq2 - mg2) * (L_qq + R_qq) * real(DB0(0, mg2, mq2));
		double term4 = 4 * (pow(mg2, 2) + (M2 - 2 * mq2) * mg2 + pow(mq2, 2)) * (sL_qq + sR_qq) / M2 * real(C0(0, M2, 0, mq2, mg2, mq2));

		result += term1 - term2 + term3 + term4;
	}
	return result;
}

double f_Z(double M2, int a, void* p) {
	struct params* fp = (struct params*)p;
	double L_qq = pow(fp->L[a], 2);
	double R_qq = pow(fp->R[a], 2);
	double result = 2 * (L_qq + R_qq);

	for (int i = 0; i < 2; i++) {
		double mg2 = pow(fp->m_g, 2);
		double mq2 = pow(fp->m_sq[i], 2);
		double S_i = pow(fp->S_ij[i][0], 2);

		double term2 = 2 * (2 * mg2 - 2 * mq2 + M2) * (L_qq * S_i + R_qq * S_i) / M2 * real(B0(0, mg2, mq2));
		double term3 = (mq2 - mg2) * (L_qq + R_qq) * real(DB0(0, mg2, mq2));

		for (int j = 0; j < 2; j++) {
			double mq2_j = pow(fp->m_sq[j], 2);

			double sL_qq = fp->sLq[a][i][j];
			double sR_qq = fp->sRq[a][i][j];

			double term1 = 2 * (2 * mg2 - mq2 - mq2_j + M2) * pow((sL_qq + sR_qq), 2) / M2 *real(B0(M2, mq2, mq2_j));
			double term4 = 4 * (pow(mg2, 2) + (M2 - mq2 - mq2_j) * mg2 + mq2 * mq2_j) * pow((sL_qq + sR_qq),2) / M2 * real(C0(0, M2, 0, mq2, mg2, mq2_j));
			result += term1 + term4;
		}

		result += term3 - term2;
	}
	return result;
}

double susy_cross(double M2, int a, int L_ID, void* p) {
	struct params* fp = (struct params*)p;

	double susy_xsec = 0;
	double C_F = 4.0 / 3;

	double beta = sqrt(1 + pow(fp->m_f, 4) / pow(M2, 2) + pow(fp->m_f, 4) / pow(M2, 2) - 2 * (pow(fp->m_f, 2) / M2 + pow(fp->m_f, 2) / M2 + (pow(fp->m_f, 2) * pow(fp->m_f, 2)) / pow(M2, 2)));
	double frac = pow(fp->alpha, 2) * fp->pi * C_F * pow(beta, 3) / (36 * M2);

	double f_g = f_gamma(M2, fp->m_g, fp->m_sq);
	double f_gZ = f_gammaZ(M2, a, fp);
	double f_Zboson = f_Z(M2, a, fp);

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			double term1 = f_g * pow(fp->e_q[a], 2) * pow(fp->e_l[L_ID], 2) * fp->delta[i][j];//f_gamma(M2, fp->m_g, fp->m_sq)*
			double term2 = f_gZ * fp->e_q[a] * fp->e_l[L_ID] * fp->delta[i][j] * (fp->sL[L_ID][i][j] + fp->sR[L_ID][i][j]) / (4 * fp->sin_thetaW * (1 - fp->sin_thetaW) * (1 - pow(fp->m_Z, 2) / M2));//f_gammaZ(M2, a, fp)*
			double term3 = f_Zboson * pow((fp->sL[L_ID][i][j] + fp->sR[L_ID][i][j]), 2) / (32 * pow(fp->sin_thetaW, 2) * pow((1 - fp->sin_thetaW), 2) * pow((1 - pow(fp->m_Z, 2) / M2), 2));//f_Z(M2, a, fp) *

			susy_xsec += frac * (term1 + term2 + term3);
		}
	}
	clearcache();

	return susy_xsec;
}*/

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

	std::map<int, double> pdfs_a = fp->pdfs[fp->imems]->xfxQ2(x[0], pow(fp->mu_F, 2));
	std::map<int, double> pdfs_b = fp->pdfs[fp->imems]->xfxQ2(x[1], pow(fp->mu_F, 2));

	if (M2 < 4 * pow(fp->m_f, 2)) {
		return 0;
	}
	else {
		double LO_xsec = 0;
		int L_ID = fp->lepton_type;
		for (int i = 0; i < 10; i++) {
			int a = abs((fp->pids[i])) % 2;

			double partonic_xsec = LO_cross(M2, a, L_ID, fp);

			LO_xsec += pdfs_a[fp->pids[i]] / x[0] * pdfs_b[-1*fp->pids[i]] / x[1] * partonic_xsec;
		}

		return LO_xsec;
	}
}

double plus_integrand(double z, double M2, int a, int L_ID, void* p) {
	struct params* fp = (struct params*)p;

	double C_F = 4.0 / 3;

	double LO_xsec = LO_cross(M2, a, L_ID, fp);
	double M2_Z1 = M2 / z;
	double LO_xsecZ1 = LO_cross(M2_Z1, a, L_ID, fp);

	double partonic_xsec = 4 * C_F * (log(1 - z) / (1 - z) *(LO_xsec - LO_xsecZ1)) + 2 * C_F * (LO_xsec - LO_xsecZ1) / (1 - z);

	return partonic_xsec;
}

double plus_corrections(double x[], size_t dim, void* p) {
	struct params* fp = (struct params*)p;

	std::map<int, double> pdfs_a = fp->pdfs[fp->imems]->xfxQ2(x[1], pow(fp->mu_F, 2));
	std::map<int, double> pdfs_b = fp->pdfs[fp->imems]->xfxQ2(x[2], pow(fp->mu_F, 2));

	double M2 = x[1] * x[2] * fp->S;
	double z = x[0];

	if (M2 < 4 * pow(fp->m_f, 2)) {
		return 0;
	}
	else {
		double plus_correction = 0;
		int L_ID = fp->lepton_type;
		double C_F = 4.0 / 3;
		for (int i = 0; i < 10; i++) {
			int a = abs((fp->pids[i])) % 2;

			double LO_xsec1 = LO_cross(M2, a, L_ID, fp);
			double corrections = 2 * C_F * LO_xsec1 * (2 * log(1 - z) / (1.0 - z) + 1.0 / (1.0 - z));

			plus_correction += pdfs_a[fp->pids[i]] / x[1] * pdfs_b[-1 * fp->pids[i]] / x[2] * fp->alpha_s / fp->pi * corrections;
		}
		return plus_correction;
	}
}

double P_qg(double z, double T_R) {
	return T_R / 2 * (pow(z, 2) + pow((1 - z), 2));
}

double Z_NLO_integrand(double x[], size_t dim, void* p) {
	struct params* fp = (struct params*)p;

	std::map<int, double> pdfs_a = fp->pdfs[fp->imems]->xfxQ2(x[1], pow(fp->mu_F, 2));
	std::map<int, double> pdfs_b = fp->pdfs[fp->imems]->xfxQ2(x[2], pow(fp->mu_F, 2));

	double M2 = x[0] * x[1] * x[2] * fp->S;
	double z = x[0];

	if (M2 < 4 * pow(fp->m_f, 2)) {
		return 0;
	}
	else {
		double NLO_Z = 0;

		int L_ID = fp->lepton_type;
		double C_F = 4.0 / 3;
		double T_R = 1.0 / 2;

		double z2 = pow(z, 2);

		double qq_term1 = -1 * (1 + z2) / (1 - z) * log(z);
		double qq_term2 = -2 * (1 + z) * log(1 - z);
		double qq_term3 = -1 * (1 + z) * log(M2 / pow(fp->mu_F, 2));

		double qg_terms = (1.0 / 2 - z + z2) * log(pow((1 - z), 2) / z) + 1.0 / 4 + 3 * z / 2 - 7 * z2 / 4 + P_qg(z, T_R) / T_R * log(M2 / pow(fp->mu_F, 2));

		for (int i = 0; i < 10; i++) {
			int a = abs((fp->pids[i])) % 2;

			double LO_xsec = LO_cross(M2, a, L_ID, fp);
			double frac = fp->alpha_s / (fp->pi) * LO_xsec * pdfs_a[fp->pids[i]] / x[1];

			double qq_term = pdfs_b[-1*fp->pids[i]] / x[2] * C_F * (qq_term1 + qq_term2 + qq_term3);
			double qg_term = 2 * pdfs_b[21] / x[2] * T_R * qg_terms;

			double partonic_xsec = frac * (qq_term + qg_term);

			//double plus_term = pdfs_a[fp->pids[i]] / x[1] * pdfs_b[-1 * fp->pids[i]] / x[2] * fp->alpha_s / fp->pi * plus_integrand(z, M2, a, L_ID, fp);

			NLO_Z += partonic_xsec; //+ plus_term;
		}

		return NLO_Z;
	}
}

double xa_xb_integrand(double x[], size_t dim, void* p) {
	struct params* fp = (struct params*)p;

	std::map<int, double> pdfs_a = fp->pdfs[fp->imems]->xfxQ2(x[0], pow(fp->mu_F, 2));
	std::map<int, double> pdfs_b = fp->pdfs[fp->imems]->xfxQ2(x[1], pow(fp->mu_F, 2));

	double M2 = x[0] * x[1] * fp->S;

	if (M2 < 4 * pow(fp->m_f, 2)) {
		return 0;
	}
	else {
		double NLO_Z1 = 0;

		int L_ID = fp->lepton_type;
		double C_F = 4.0 / 3;

		double NLO_terms = pow(fp->pi, 2) / 3 - 4 + 3.0 / 2 * log(M2 / pow(fp->mu_F, 2));

		for (int i = 0; i < 10; i++) {
			int a = abs((fp->pids[i])) % 2;

			double LO_xsec = LO_cross(M2, a, L_ID, fp);

			//double susy_xsec = fp->alpha_s / fp->pi * susy_cross(M2, a, L_ID, fp);

			//double plus_terms = fp->alpha_s / fp->pi * (LO_xsec * 4 * C_F * -1.0 / 2 * pow(log(1 - tau), 2) - 2 * C_F * LO_xsec * log(1 - tau));

			double partonic_xsec = LO_xsec * (1 + fp->alpha_s / (fp->pi) * C_F * NLO_terms);// + susy_xsec;

			NLO_Z1 += pdfs_a[fp->pids[i]] / x[0] * pdfs_b[-1 * fp->pids[i]] / x[1] * partonic_xsec;
		}

		return NLO_Z1;
	}
}

int main(int argc, char* argv[]) {

	struct params p;
	struct params init;
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
				p.sLq[i][j][k] = p.L[i] * p.S_ij[j][0] * p.S_ij[k][0];
				p.sRq[i][j][k] = p.R[i] * p.S_ij[j][1] * p.S_ij[k][1];
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

	cout << "sR is of type" << p.lR[1] << endl;

	time_t start_time = time(NULL);

	const string setname = argv[1];
	const LHAPDF::PDFSet set(setname);
  const size_t nmem = set.size()-1;

	const vector<LHAPDF::PDF*> pdf_sets = set.mkPDFs();

	p.pdfs = pdf_sets;
	//p.pdf = mkPDF(setname, imem);
	ofstream myfile;
	myfile.open("time_cpp.dat");
	myfile << "pdf_nr     time_elapsed\n";

	for (size_t imem = 0; imem <= nmem; imem++) {
		p.imems = imem;

		vector<int> pids = p.pdfs[imem]->flavors();

		p.pids = pids;

		std::map<int, double> pdfs = p.pdfs[imem]->xfxQ2(0.25, pow(2.0, 2));

		/*for (int a = 0; a < 10; a++) {
			cout << "pdf value at pid " << pids[a] << " is: " << pdfs[pids[a]] << endl;
		}*/

		for (int i = 0; i < 1; i++) {
			double min_mf = 140;
			double max_mf = 350;
			double dm = (max_mf - min_mf) / 10;
			p.m_f = min_mf + i * dm;
			p.mu_F = p.m_f;
			p.mu_R = p.m_f;
			p.alpha_s = p.pdfs[imem]->alphasQ2(pow(p.mu_R, 2));
			p.lepton_type = 1;

			double xsec = 0;

			double tau = 4 * pow(p.m_f, 2)/p.S;

			size_t dim1 = 2;
			size_t dim2 = 3;

			double x0[dim1] = { tau, tau };
			double x1[dim1] = { 1.0, 1.0 };

			double x0_NLO[dim2] = { tau, tau, tau };
			double x1_NLO[dim2] = { 1.0, 1.0, 1.0 };

			double x0_plus[dim2] = { 0, tau, tau };
			double x1_plus[dim2] = { tau, 1.0, 1.0 };

			const gsl_rng_type* T;
			gsl_rng* r;

			gsl_monte_function G = { &xa_xb_integrand, dim1, &p };
			gsl_monte_function H = { &Z_NLO_integrand, dim2, &p };
			gsl_monte_function plus = { &plus_corrections, dim2, &p };

			size_t calls = 100000;

			gsl_rng_env_setup();

			T = gsl_rng_default;
			r = gsl_rng_alloc(T);

			double res1 = 0;
			double err1 = 0;

			double res2 = 0;
			double err2 = 0;

			double res_plus = 0;
			double err_plus = 0;

			{
				printf("starting integration: \n");
				gsl_monte_vegas_state* s = gsl_monte_vegas_alloc(dim1);
				gsl_monte_vegas_state* t = gsl_monte_vegas_alloc(dim2);
				gsl_monte_vegas_state* u = gsl_monte_vegas_alloc(dim2);

				//ltini();
				gsl_monte_vegas_integrate(&G, x0, x1, dim1, 10000, r, s, &res1, &err1);

				do
				{
					gsl_monte_vegas_integrate(&G, x0, x1, dim1, calls / 5, r, s,
						&res1, &err1);
					//printf("result = % .6e sigma = % .6e chisq/dof = %.3e\n", res1, err1, gsl_monte_vegas_chisq(s));
				} while (fabs(gsl_monte_vegas_chisq(s) - 1) > 0.5);

				gsl_monte_vegas_free(s);

				printf("done with xa_xb! \n");

				gsl_monte_vegas_integrate(&H, x0_NLO, x1_NLO, dim2, 10000, r, t, &res2, &err2);

				do
				{
					gsl_monte_vegas_integrate(&H, x0_NLO, x1_NLO, dim2, calls / 5, r, t,
						&res2, &err2);
					//printf("result = % .6e sigma = % .6e chisq/dof = %.3e\n", res, err, gsl_monte_vegas_chisq(s));
				} while (fabs(gsl_monte_vegas_chisq(t) - 1) > 0.5);

				gsl_monte_vegas_free(t);

				printf("done with Z integral!\n");

				gsl_monte_vegas_integrate(&plus, x0_plus, x1_plus, dim2, 10000, r, u, &res_plus, &err_plus);

				do
				{
					gsl_monte_vegas_integrate(&plus, x0_plus, x1_plus, dim2, calls / 5, r, u,
						&res_plus, &err_plus);
					//printf("result = % .6e sigma = % .6e chisq/dof = %.3e, while term: %.3e\n", res_plus, err_plus, gsl_monte_vegas_chisq(u), fabs(gsl_monte_vegas_chisq(u) - 1));
				} while (fabs(gsl_monte_vegas_chisq(u) - 1) > 0.5);

				gsl_monte_vegas_free(u);
			}

			cout << "NLO xsec res for xa_xb integration: " << res1*0.38938e-3 << " and for the z integration: " << res2*0.38938e-3 << endl;
			cout << "and the sum is: " << (res1+res2)*0.38938e-3 << endl;

			xsec = res1 + res2 - res_plus;

			gsl_monte_function LO = { &LO_integrand, dim1, &p };

			double xsec_LO = 0;
			double err_LO = 0;

			{
				gsl_monte_vegas_state* v = gsl_monte_vegas_alloc(dim1);

				gsl_monte_vegas_integrate(&LO, x0, x1, dim1, 10000, r, v, &xsec_LO, &err_LO);
				size_t calls = 100000;

				do
				{
					gsl_monte_vegas_integrate(&LO, x0, x1, dim1, calls / 5, r, v, &xsec_LO, &err_LO);
				} while (fabs(gsl_monte_vegas_chisq(v) - 1) > 0.5);

				gsl_monte_vegas_free(v);
			}

			gsl_rng_free(r);

			xsec *= 0.38938e-3*1e15;
			xsec_LO *= 0.38938e-3*1e15;
			printf("Final NLO res after summing up Parton ID's: %.5e with selectron mass: %.2f\n", xsec, p.m_f);
			printf("Final LO res after summing up Parton ID's: %.5e with selectron mass: %.2f\n", xsec_LO, p.m_f);
		}
		time_t end_time = time(NULL);
		cout << "time: " << end_time << "\n";
		myfile << imem+1 << " " << end_time-start_time << "\n";
		cout << "percent done of members:" << (imem+1)/nmem << "\n";

	}
	myfile.close();

	//delete p.pdfs;

	return 0;
}
