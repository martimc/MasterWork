#include <iostream>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <random>
#include <cmath>
#include <complex>
//#include "clooptools.h"

using namespace std;

struct test {
	double data;
	double alpha[2];
};

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
	double *m_f; //= 114.8;
	double m_g = 2e3;
	double S = pow(14e3, 2);
	double T_f[2], e_q[2], L[2], R[2], lL[2], lR[2], m_sq[2];
	double S_ij[2][2], delta[2][2], sL[2][2][2], sR[2][2][2], sLq[2][2][2], sRq[2][2][2];
	double *M2;
	double *z;
	int lepton_type;
	double mu_F;
	double mu_R;
	double *alpha_s;
	int n;
	double *LO_xsec;
	double *susy_xsec;
	int *pid;
	double *Z1_xsec;
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
}*/

double LO_xsec(struct params* fp, double M2, int L_ID, int a){
	double LO_out = 0;

	double beta = sqrt(1 + pow(fp->m_f[0], 4) / pow(M2, 2) + pow(fp->m_f[1], 4) / pow(M2, 2) - 2 * (pow(fp->m_f[0], 2) / M2 + pow(fp->m_f[1], 2) / M2 + (pow(fp->m_f[0], 2) * pow(fp->m_f[1], 2)) / pow(M2, 2)));
	double frac = pow(fp->alpha, 2) * fp->pi * pow(beta, 3) / (9 * M2);

	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < 1; j++) {
			double term1 = pow(fp->e_q[a], 2) * pow(fp->e_l[L_ID], 2) * fp->delta[i][j];
			double term2 = fp->e_q[a] * fp->e_l[L_ID] * fp->delta[i][j] * (fp->L[a] + fp->R[a]) * (fp->sL[L_ID][i][j] + fp->sR[L_ID][i][j]) / (4 * fp->sin_thetaW * (1 - fp->sin_thetaW) * (1 - pow(fp->m_Z, 2) / M2));
			double term3 = (pow(fp->L[a], 2) + pow(fp->R[a], 2)) * pow((fp->sL[L_ID][i][j] + fp->sR[L_ID][i][j]), 2) / (32 * pow(fp->sin_thetaW, 2) * pow((1 - fp->sin_thetaW), 2) * pow((1 - pow(fp->m_Z, 2) / M2), 2));

			LO_out += frac * (term1 + term2 + term3);
		}
	}
	return LO_out;
}

double P_qg(double z, double T_R) {
	return T_R / 2 * (pow(z, 2) + pow((1 - z), 2));
}

extern "C" {

void LO_cross(struct params* fp ) {
	const double* M2 = fp->M2;
	const double* z = fp->z;
	int L_ID = fp->lepton_type;

	for (int k = 0; k < fp->n; k++){

		if (z[k]*M2[k] < pow(fp->m_f[0]+fp->m_f[1],2)){
			for (int i = 0; i < 5; i++){
				fp->LO_xsec[5*k+i] = 0;
			}
		}
		else{
			for (int g = 0; g < 5; g++){
				int a = abs((fp->pid[g])) % 2;

				double beta = sqrt(1 + pow(fp->m_f[0], 4) / pow(M2[k], 2) + pow(fp->m_f[1], 4) / pow(M2[k], 2) - 2 * (pow(fp->m_f[0], 2) / M2[k] + pow(fp->m_f[1], 2) / M2[k] + (pow(fp->m_f[0], 2) * pow(fp->m_f[1], 2)) / pow(M2[k], 2)));
				double frac = pow(fp->alpha, 2) * fp->pi * pow(beta, 3) / (9 * M2[k]);

				for (int i = 0; i < 1; i++) {
					for (int j = 0; j < 1; j++) {
						double term1 = pow(fp->e_q[a], 2) * pow(fp->e_l[L_ID], 2) * fp->delta[i][j];
						double term2 = fp->e_q[a] * fp->e_l[L_ID] * fp->delta[i][j] * (fp->L[a] + fp->R[a]) * (fp->sL[L_ID][i][j] + fp->sR[L_ID][i][j]) / (4 * fp->sin_thetaW * (1 - fp->sin_thetaW) * (1 - pow(fp->m_Z, 2) / M2[k]));
						double term3 = (pow(fp->L[a], 2) + pow(fp->R[a], 2)) * pow((fp->sL[L_ID][i][j] + fp->sR[L_ID][i][j]), 2) / (32 * pow(fp->sin_thetaW, 2) * pow((1 - fp->sin_thetaW), 2) * pow((1 - pow(fp->m_Z, 2) / M2[k]), 2));

						fp->LO_xsec[5*k+g] += frac * (term1 + term2 + term3);
					}
				}
			}
		}
	}
}

void Z1_cross( struct params* fp ) {
	const double* M2 = fp->M2;

	int L_ID = fp->lepton_type;
	double C_F = 4.0/3;

	for (int i = 0; i < fp->n; i++){

		if (M2[i] < pow(fp->m_f[0]+fp->m_f[1],2)){
			for (int j = 0; j < 5; j++){
				fp->Z1_xsec[5*i+j] = 0;
			}
		}
		else{
			double NLO_terms = pow(fp->pi, 2) / 3 - 4 + 3.0 / 2 * log(M2[i] / pow(fp->mu_F, 2));
			for (int j = 0; j < 5; j++){
				int a = abs((fp->pid[j])) % 2;

				double sigma_LO = LO_xsec(fp, M2[i], L_ID, a);
				fp->Z1_xsec[5*i+j] = sigma_LO*(1 + fp->alpha_s[i] / fp->pi * C_F * NLO_terms);
			}
		}
	}
}

/*void susy_cross(struct params* fp) {

	int a = fp->a;
	const double* M2 = fp->M2;
	int L_ID = fp->lepton_type;

	double C_F = 4.0 / 3;

	for (int k = 0; k < fp->n; k++){
		double beta = sqrt(1 + pow(fp->m_f, 4) / pow(M2[k], 2) + pow(fp->m_f, 4) / pow(M2[k], 2) - 2 * (pow(fp->m_f, 2) / M2[k] + pow(fp->m_f, 2) / M2[k] + (pow(fp->m_f, 2) * pow(fp->m_f, 2)) / pow(M2[k], 2)));
		double frac = pow(fp->alpha, 2) * fp->pi * C_F * pow(beta, 3) / (36 * M2[k]);

		ltini();

		for (int i = 0; i < 1; i++) {
			for (int j = 0; j < 1; j++) {
				double term1 = f_gamma(M2[k], fp->m_g, fp->m_sq) * pow(fp->e_q[a], 2) * pow(fp->e_l[L_ID], 2) * fp->delta[i][j];
				double term2 = f_gammaZ(M2[k], a, fp) * fp->e_q[a] * fp->e_l[L_ID] * fp->delta[i][j] * (fp->sL[L_ID][i][j] + fp->sR[L_ID][i][j]) / (4 * fp->sin_thetaW * (1 - fp->sin_thetaW) * (1 - pow(fp->m_Z, 2) / M2[k]));
				double term3 = f_Z(M2[k], a, fp) * pow((fp->sL[L_ID][i][j] + fp->sR[L_ID][i][j]), 2) / (32 * pow(fp->sin_thetaW, 2) * pow((1 - fp->sin_thetaW), 2) * pow((1 - pow(fp->m_Z, 2) / M2[k]), 2));

				fp->susy_xsec[k] += frac * (term1 + term2 + term3);
			}
		}
		clearcache();
	}
}*/

}
