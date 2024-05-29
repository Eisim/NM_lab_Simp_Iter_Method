#include <iostream>
#include <vector>
#include <cmath>

#include <iostream>
#include <fstream>
const double PI = 3.1415926535897932384626433832795028841971693993751058209;

using namespace std;
string path_to_save = "./data/";

int iterations = 0;
int global_N_max = 1;
double global_eps = 1;
double global_eps_method = 1;
double global_cur_eps = 0;
double global_cur_eps_method = 0;
int global_eps_exit = 0;
int global_eps_method_exit = 0;


extern "C" __declspec(dllexport) float get_iteration() {
	float cur_percent = 0;
	float eps_percent = min({ (float)(global_eps_exit * global_eps / global_cur_eps) ,1.f });
	float eps_method_percent = min({ (float)(global_eps_method_exit * global_eps_method / global_cur_eps_method),  1.f});
	cur_percent = max({ (float)iterations / (float)global_N_max , eps_percent, eps_method_percent });
	return cur_percent;
}




double u(double x, double y) {
	return exp(pow(sin(PI * x * y), 2));
}
double f(double x, double y) {

	return -u(x, y) * (pow(sin(2 * PI * x * y), 2) + 2*cos(2 * PI * x * y)) * PI * PI*(y * y + x * x);
}
template<class T>
class csvWriter {
	string file_name;

public:
	static void write(string file_name, string path, vector<string> columns, vector<vector<T>> data = {}) {
		ofstream out;
		out.open(path + file_name);
		if (out.is_open())
		{
			if (columns.size() != 0) {
				//add columns name
				for (int i = 0; i < columns.size(); i++) {
					out << columns[i];
					if (i < columns.size() - 1) {
						out << ',';
					}
				}
				out << '\n';
			}
			//add data
			for (auto line : data) {
				for (int i = 0; i < line.size();i++) {
					out << line[i];
					if (i < line.size() - 1) {
						out << ',';
					}
				}
				out << '\n';
			}

		}
		out.close();
	}
};

class Solution {
	int n, m;
	double a, b, c, d, h, k;

	bool check_for_multiplicity(int n, int m) {
		return !((n % 2) && (m % 2));
	}
	void fill_border_conditions() {
		int shift = 0;

		//mu_1
		for (int j = 0; j <= m; j++) {
			xy_1[0][j] = u(x[0], y[j]);
		}
		//mu_2
		for (int j = m / 2; j <= m; j++) {
			xy_1[n / 2][j] = u(x[n / 2], y[j]);
		}
		//mu_3
		for (int j = 0; j <= m / 2; j++) {
			xy_2[n / 2 - 1][j] = u(x[n], y[j]);
		}
		//mu_4
		vector<vector<double>>* cur_matrix = &xy_1;
		for (int i = 0; i <= n; i++) {
			if (i > n / 2) {
				cur_matrix = &xy_2;
				shift = n / 2 + 1;
			}
			(*cur_matrix)[i - shift][0] = u(x[i], 0);
		}
		//mu_5
		for (int i = 0; i < n / 2; i++) {
			xy_2[i][m / 2] = u(x[i + n / 2 + 1], y[m / 2]);
		}
		//mu_6
		for (int i = 0; i <= n / 2; i++) {
			xy_1[i][m] = u(x[i], y[m]);
		}

	}
public:
	double eps = 0;
	double eps_method = 0;
	double r_max = 0;

	//число итераций дл€ достижени€ точности V
	double z0_norm_2 = 0;
	int s_met = 0;
	int n_sch = 0, m_sch = 0;
	//число итераций дл€ достижени€ точности ^

	//переменные дл€ теоретических расчЄтов V
	double r2_norm = 0;
	double r2_0 = 0;
	double r2_N = 0;

	//переменные дл€ теоретических расчЄтов ^

	int res_N = 0;
	vector<double> x, y;
	vector<vector<double>> xy_1, xy_2, r_1, r_2, dif_matrix_1, dif_matrix_2;
	double argmax_dif_x = 0, argmax_dif_y = 0, max_dif;
	double tau;
	Solution(double a, double b, double c, double  d, int n, int m) {
		if (!check_for_multiplicity(n, m) || n<=0 ||m<=0) {
			throw exception("—етка не накладываетс€ на область");
		}
		this->n = n;
		this->m = m;
		this->a = a; 
		this->b = b; 
		this->c = c; 
		this->d = d;
		this->h = (b - a) / n;
		this->k = (d - c) / m;

		//вспомогательные векторы (значени€ x, y на сетке)
		for (int i = 0; i <= n; i++) {
			x.push_back(a + i * h);
		}
		for (int i = 0; i <= m; i++) {
			y.push_back(c + i * k);
		}

		xy_1 = vector(n / 2 + 1, vector<double>(m + 1, 0));
		xy_2 = vector(n / 2, vector<double>(m / 2 + 1, 0));

		r_1 = vector(n / 2 + 1, vector<double>(m + 1, 0));
		r_2 = vector(n / 2, vector<double>(m / 2 + 1, 0));

		dif_matrix_1 = vector(n / 2 + 1, vector<double>(m + 1, 0));
		dif_matrix_2 = vector(n / 2, vector<double>(m / 2 + 1, 0));
	}

	void calc_params(double eps, double M_max, double M_min) {
		double tmp = 0;
		
		double max_u_4x = 5295.707241688835;
		n_sch = sqrt( (max_u_4x)/(12*4*eps/3))+1;
		n_sch = (n_sch % 2) ? n_sch + 1 : n_sch;
		m_sch = n_sch;

		vector<double> x_sch, y_sch;
		double h_sch = 1 / n_sch, k_sch = 1 / m_sch;
		for (int i = 0; i <= n_sch; i++) {
			x_sch.push_back(a + i * h_sch);
		}
		for (int i = 0; i <= m_sch; i++) {
			y_sch.push_back(c + i * k_sch);
		}

		for (int i = 1; i < (n_sch)/2; i++){
			for (int j = 1; j < m_sch; j++){
				tmp += u(x[i], y[j]) * u(x[i], y[j]);
			}
		}
		for (int i = 1; i < n_sch/2; i++) {
			for (int j = 1; j < xy_2[0].size(); j++) {
				tmp += u(x[i + n / 2 + 1], y[j]) * u(x[i + n / 2 + 1], y[j]);
			}
		}

		z0_norm_2 = sqrt(tmp);
		double eps_met = eps / 3;
		double mu_A = M_max / M_min;
		s_met = log(eps_met /z0_norm_2)/log((mu_A - 1)/(mu_A + 1));
		
		csvWriter<int>::write("Needed_params.csv", path_to_save, {"n: ","m: ","s_met: "}, {{n_sch, m_sch, s_met}});
	}

	void show_matrix(const vector<vector<double>>& xy_1, const vector<vector<double>>& xy_2) {
		std::cout << "ѕерва€ часть:\n";
		for (int j = xy_1[0].size() - 1; j >= 0; j--) {
			for (size_t i = 0; i < xy_1.size(); i++) {
				std::cout << xy_1[i][j] << ' ';
			}
			std::cout << '\n';
		}
		std::cout << "¬тора€ часть:\n";
		for (int j = xy_2[0].size() - 1; j >= 0; j--) {
			for (size_t i = 0; i < xy_2.size(); i++) {
				std::cout << xy_2[i][j] << ' ';
			}
			std::cout << '\n';
		}
	}
	void r_s() {
		// Ќев€зка:

		size_t n = x.size() - 1;
		size_t m = y.size() - 1;
		double h = x[1] - x[0];
		double k = y[1] - y[0];
		double h2 = 1 / h / h;
		double k2 = 1 / k / k;
		// Ќев€зка:
		double a_coef = -2 * (h2 + k2);
		for (int i = 1; i < n / 2; i++) {
			for (int j = 1; j < m; j++) {
				r_1[i][j] = a_coef * xy_1[i][j] + h2 * (xy_1[i - 1][j] + xy_1[i + 1][j]) + k2 * (xy_1[i][j + 1] +  xy_1[i][j - 1]) + f(x[i], y[j]);
				r_max = fmax(r_max, fabs(r_1[i][j]));
				r2_norm += r_1[i][j] * r_1[i][j];
			}
		}
		for (int i = 1; i < n / 2 - 1; i++) {
			for (int j = 1; j < m / 2; j++) {
				r_2[i][j] = a_coef * xy_2[i][j] + h2 * (xy_2[i - 1][j] + xy_2[i + 1][j]) + k2 *(xy_2[i][j + 1] + xy_2[i][j - 1]) + f(x[i + n / 2 + 1], y[j]);
				r_max = fmax(r_max, fabs(r_2[i][j]));
				r2_norm += r_2[i][j] * r_2[i][j];
			}
		}
		// нев€зка на границе двух частей: xy_1 и xy_2
		int i = n / 2;
		for (int j = 1; j < m / 2; j++) {
			r_1[i][j] = a_coef * xy_1[i][j] + h2 * (xy_1[i - 1][j] +  xy_2[0][j]) +
				k2 * (xy_1[i][j + 1] + xy_1[i][j - 1]) +
				f(x[i], y[j]);
			r_max = fmax(r_max, fabs(r_1[i][j]));
			r2_norm += r_2[i][j] * r_1[i][j];
		}
		i = 0;
		for (int j = 1; j < m / 2; j++) {
			r_2[i][j] = a_coef * xy_2[i][j] + h2 * (xy_1[n / 2][j] +  xy_2[i + 1][j]) +
				k2 * (xy_2[i][j + 1] +  xy_2[i][j - 1]) +
				f(x[n / 2 + 1], y[j]);
			r_max = fmax(r_max, fabs(r_2[i][j]));
			r2_norm += r_2[i][j] * r_2[i][j];
		}
		r2_norm = sqrt(r2_norm);
	}
	void step(vector<vector<double>>& xy, const vector<vector<double>>& r_s, double tau, vector<vector<double>>& d_m, double shift) {
		//„исленное решение
		// [ x^(s+1) - x^(s) ]/tau +Ax^(s)  = b
		// x^(s+1) = tau*(b - Ax^(s)) + x^(s)  
		// x^(s+1) = x^(s) - tau*(r^(s))
		double tmp,loc_eps_method = 0;
#pragma omp parallel for collapse(2) reduction(max:eps, max_dif, loc_eps_method) // collapse(2) дл€ параллелизации обоих циклов
		for (int i = 0; i < xy.size(); i++) {
			for (int j = 0; j < xy[0].size(); j++) {
				xy[i][j] = xy[i][j] - tau * r_s[i][j] * (-1);//т.к. матрица не €вл€етс€ положительно определенной
				eps = fmax(eps,fabs( u(x[i+shift],y[j]) - xy[i][j]));
				d_m[i][j] = fabs(u(x[i+shift], y[j]) - xy[i][j]);
				if (d_m[i][j] > max_dif) {
					max_dif = d_m[i][j];
					argmax_dif_x = x[i+ shift];
					argmax_dif_y = y[j+ shift];
				}
				loc_eps_method = fmax(loc_eps_method,abs(tau*r_s[i][j]));
			}
		}
		eps_method = loc_eps_method;
	}

	double lambd(int l, int s) {
		return -(4 / h / h * pow(sin(PI * l / 2 / n), 2) + 4 / k / k * pow(sin(PI*s/2/m), 2));
	}

	pair<double, double> l_min_max() {
		double l_min = abs(lambd(1,1));
		double l_max = abs(lambd(n-1,m-1));
		return { l_min, l_max};
	}

	void process(int N_max,double eps_user, double accur_user, int accur_exit, int eps_exit) {
		//»менно дл€ задачи ƒирихле уравнени€ ѕуассона
		pair<double, double> l_m_M = l_min_max();

		double M_max = l_m_M.second;
		double M_min = l_m_M.first;

		tau = 2 / (M_min+M_max);
		calc_params(eps_user,M_max, M_min);
		//«аполнение сетки граничными услови€ми:	
		fill_border_conditions();
		r_s();
		r2_0 = r2_norm;
		eps_method = eps_user;
		iterations = 0;
		for (iterations; iterations < N_max; iterations++) {
			r_max = 0;
			r2_norm = 0;
			eps = 0;
			eps_method = accur_user;
			max_dif = 0;
			r_s();
			step(xy_1, r_1, tau,dif_matrix_1,  0);
			step(xy_2, r_2, tau,dif_matrix_2,  n/2+1);
			global_cur_eps_method = eps_method;
			global_cur_eps = eps;
			if (eps < eps_user && eps_exit) {
				break;
			}
			else if (eps_method < accur_user && accur_exit) {
				break;
			}

		}
		r_max = 0;
		r_s();
		r2_N = r2_norm;
		this->res_N = iterations;
		iterations = N_max;
	}
};
extern "C" __declspec(dllexport) void calc_params(int n, int m, double eps, double eps_method) {

}
extern "C" __declspec(dllexport)  void main_f(int n, int m, int N_max, double eps,double accur, int accur_exit, int eps_exit) {
	double a = 0., b = 1., c = 0., d = 1.;
	
	iterations = 0;
	global_N_max = N_max;
	global_eps = eps;
	global_eps_method = accur;
	global_eps_exit = eps_exit;
	global_eps_method_exit = accur_exit;


	Solution sol(a,b,c,d, n, m);
	sol.process(N_max,eps,accur,accur_exit,eps_exit);


	csvWriter<double>::write("r_part1.csv", path_to_save, {}, sol.r_1);
	csvWriter<double>::write("r_part2.csv", path_to_save, {}, sol.r_2);

	csvWriter<double>::write("dif_part1.csv", path_to_save, {}, sol.dif_matrix_1);
	csvWriter<double>::write("dif_part2.csv", path_to_save, {}, sol.dif_matrix_2);

	csvWriter<double>::write("v_part1.csv", path_to_save, {}, sol.xy_1);
	csvWriter<double>::write("v_part2.csv", path_to_save, {}, sol.xy_2);

	csvWriter<double>::write("extra_info.csv", path_to_save, { "макс. обща€ погрешность: ","макс. нев€зка: ","макс. точность метода: ","„исло шагов:","макс. |u(x;y) - v(x;y)| = ","при x = ","при y = ","ѕараметр tau: ","n: ","m: "}, {{sol.eps,sol.r_max,sol.eps_method,(double)sol.res_N,sol.max_dif, sol.argmax_dif_x,sol.argmax_dif_y,sol.tau, (double)n, (double)m}});
	
	csvWriter<double>::write("theoretical_info.csv", path_to_save, { "Ќорма нев€зки на нулевом шаге: ","норма нев€зки на последнем шаге: " }, { {sol.r2_0,sol.r2_N} });

	global_eps = 0.;
	global_eps_method = 0.;
}