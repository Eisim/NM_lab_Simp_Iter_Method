#include <iostream>
#include <vector>
#include <cmath>

const double PI = 3.1415926535897932384626433832795028841971693993751058209;

using namespace std;

double u(double x, double y) {
	return exp(pow(sin(PI * x * y), 2));
}
double f(double x, double y) {
	// -u^{''}_{xx} - u^{''}_{yy}
	//return -(u(x, y) * PI * PI * y * y * (pow(sin(2 * PI * x * y), 2)) + 2 * cos(2 * PI * x * y)) - (u(x, y) * PI * PI * x * x * (pow(sin(2 * PI * x * y), 2)) + 2 * cos(2 * PI * x * y));
	//return -(  (pow(u(x,y),2)*sin(PI*4*x*y) *pow(PI*y,3) )   + (pow(u(x, y), 2) * sin(PI * 4 * x * y) * pow(PI * x, 3)));
	return -u(x, y) * (pow(sin(2 * PI * x * y), 2) + 2*cos(2 * PI * x * y)) * PI * PI*(y * y + x * x);
}

class Solution {
	int n, m;
	double a, b, c, d, h, k;


	vector<double> x, y;
	vector<vector<double>> xy_1, xy_2, r_1, r_2, debug_matrix_1, debug_matrix_2;

	//Переменные для оценки работы метода
	double eps = 0;
	double eps_method = 0;
	double r_max = 0;
	

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

	Solution(double a, double b, double c, double  d, int n, int m) {
		if (!check_for_multiplicity(n, m)) {
			throw exception("Сетка не накладывается на область");
		}
		this->n = n;
		this->m = m;
		this->a = a; 
		this->b = b; 
		this->c = c; 
		this->d = d;
		this->h = (b - a) / n;
		this->k = (d - c) / m;

		//вспомогательные векторы (значения x, y на сетке)
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

		debug_matrix_1 = vector(n / 2 + 1, vector<double>(m + 1, 0));
		debug_matrix_2 = vector(n / 2, vector<double>(m / 2 + 1, 0));
	}


	void show_matrix(const vector<vector<double>>& xy_1, const vector<vector<double>>& xy_2) {
		std::cout << "Первая часть:\n";
		for (int j = xy_1[0].size() - 1; j >= 0; j--) {
			for (size_t i = 0; i < xy_1.size(); i++) {
				std::cout << xy_1[i][j] << ' ';
			}
			std::cout << '\n';
		}
		std::cout << "Вторая часть:\n";
		for (int j = xy_2[0].size() - 1; j >= 0; j--) {
			for (size_t i = 0; i < xy_2.size(); i++) {
				std::cout << xy_2[i][j] << ' ';
			}
			std::cout << '\n';
		}
	}
	void r_s() {
		// Невязка:

		//a_coef = -2*(1/h/h+1/k/k)
		//r^(s) = Ax^(s) - b
		//r^(s)[i][j] = a v[i][j] +1/h/h * v[i-1][j] + 1/h/h * v[i+1][j] + 1/k/k * v[i][j+1] + 1/k/k * v[i][j-1] + f[i][j] 

		size_t n = x.size() - 1;
		size_t m = y.size() - 1;
		double h = x[1] - x[0];
		double k = y[1] - y[0];
		double h2 = 1 / h / h;
		double k2 = 1 / k / k;
		// Невязка:

			//a_coef = -2*(1/h/h+1/k/k)
			//r^(s) = Ax^(s) - b
			//r^(s)[i][j] = a v[i][j] +1/h/h * v[i-1][j] + 1/h/h * v[i+1][j] + 1/k/k * v[i][j+1] + 1/k/k * v[i][j-1] + f[i][j] 
		double a_coef = -2 * (h2 + k2);

		for (int i = 1; i < n / 2; i++) {
			for (int j = 1; j < m; j++) {
				r_1[i][j] = a_coef * xy_1[i][j] + h2 * xy_1[i - 1][j] + h2 * xy_1[i + 1][j] + k2 * xy_1[i][j + 1] + k2 * xy_1[i][j - 1] + f(x[i], y[j]);
				r_max = fmax(r_max, fabs(r_1[i][j]));
			}
		}
		for (int i = 1; i < n / 2 - 1; i++) {
			for (int j = 1; j < m / 2; j++) {
				r_2[i][j] = a_coef * xy_2[i][j] + h2 * xy_2[i - 1][j] + h2 * xy_2[i + 1][j] + k2 * xy_2[i][j + 1] + k2 * xy_2[i][j - 1] + f(x[i + n / 2 + 1], y[j]);
				r_max = fmax(r_max, fabs(r_2[i][j]));
			}
		}
		// невязка на границе двух частей: xy_1 и xy_2
		int i = n / 2;
		for (int j = 1; j < m / 2; j++) {
			r_1[i][j] = a_coef * xy_1[i][j] + h2 * xy_1[i - 1][j] + h2 * xy_2[0][j] +
				k2 * xy_1[i][j + 1] + k2 * xy_1[i][j - 1] +
				f(x[i], y[j]);
			r_max = fmax(r_max, fabs(r_1[i][j]));
		}
		i = 0;
		for (int j = 1; j < m / 2; j++) {
			r_2[i][j] = a_coef * xy_2[i][j] + h2 * xy_1[n / 2][j] + h2 * xy_2[i + 1][j] +
				k2 * xy_2[i][j + 1] + k2 * xy_2[i][j - 1] +
				f(x[n / 2 + 1], y[j]);
			r_max = fmax(r_max, fabs(r_2[i][j]));
		}
	}
	void step(vector<vector<double>>& xy, const vector<vector<double>>& r_s, double tau, vector<vector<double>>& d_m, double shift) {
		//Численное решение
		// [ x^(s+1) - x^(s) ]/tau +Ax^(s)  = b
		// x^(s+1) = tau*(b - Ax^(s)) + x^(s)  
		// x^(s+1) = x^(s) - tau*(r^(s))
		// Можно оптимизировать(не считать значения на границе)
		
		double tmp,loc_eps_method = 0;
		for (int i = 0; i < xy.size(); i++) {
			for (int j = 0; j < xy[0].size(); j++) {
				xy[i][j] = xy[i][j] - tau * r_s[i][j] * (-1);//т.к. матрица не является положительно определенной
				eps = fmax(eps,fabs( u(x[i+shift],y[j]) - xy[i][j]));
				d_m[i][j] = fabs(u(x[i+shift], y[j]) - xy[i][j]);
				loc_eps_method = fmax(loc_eps_method,abs(tau*r_s[i][j]));
			}
		}
		eps_method = loc_eps_method;
	}

	void process(int N_max,double eps_user) {

		//Именно для задачи Дирихле уравнения Пуассона
		double M_max = 4 * (1 / h / h + 1 / k / k);
		double M_min = 0;
		double tau = 2 / M_max*0.5;  //Временное значение для tau( Нужно оптимизировать его, а именно найти лучшую оценку для собственных чисел)


		//Заполнение сетки граничными условиями:	
		fill_border_conditions();
		
		eps_method = eps_user;
		int iterations = 0;
		for (iterations; iterations < N_max ; iterations++) {

			r_max = 0;
			eps = 0;
			eps_method = eps_user;
			r_s();
			step(xy_1, r_1, tau,debug_matrix_1,  0);
			step(xy_2, r_2, tau,debug_matrix_2,  n/2+1);
			if (iterations % (N_max/5) == 0) {
				cout << "Шаг " << 1 + iterations << "):\n";
				cout << "Норма невязки:" << r_max << "\nПогрешность:" << eps << "\nПогрешность метода:" << eps_method << endl;
				//show_matrix(r_1, r_2);
				

			}
		}
		
		cout << "ИТОГ:\n";
		//show_matrix(xy_1, xy_2);
		cout << "Норма невязки:" << r_max << "\nПогрешность:" << eps << "\nПогрешность метода:" << eps_method<<endl;
		cout << "Количество итераций:" << iterations;
	}
};






int main() {
	system("chcp 1251");
	int n=12, m = 12;
	double a = 0., b = 1., c = 0., d = 1.;
	int N_max = 40000;
	Solution sol(a,b,c,d, n, m);
	sol.process(N_max,1e-6);

	return 0;
}