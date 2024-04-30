#include <iostream>
#include <vector>
#include <cmath>

const double PI = 3.1415926535897932384626433832795028841971693993751058209;

using namespace std;

bool check_for_multiplicity(int n, int m) {
	return !((n % 2) && (m % 2));
}

double u(double x, double y) {
	return exp(pow(sin(PI  * x * y), 2));
}
double f(double x, double y) {
	// -u^{''}_{xx} - u^{''}_{yy}
	return -(u(x,y)*PI*PI *y*y*( pow( sin(2*PI*x*y), 2)) + 2*cos(2*PI*x*y) ) - (u(x, y) * PI * PI * x * x * (pow(sin(2 * PI * x * y), 2)) + 2 * cos(2 * PI * x * y));
}

void fill_border_conditions(vector <vector<double>>& m_1, vector < vector<double>>& m_2,vector<double>&x,vector<double>&y,int n, int m) {
	int shift = 0;

	//mu_1
	for (int j = 0; j <= m; j++) {
		m_1[0][j] = u(x[0], y[j]);
	}
	//mu_2
	for (int j = m / 2; j <= m; j++) {
		m_1[n / 2][j] = u(x[n / 2], y[j]);
	}
	//mu_3
	for (int j = 0; j <= m / 2; j++) {
		m_2[n/2-1][j] = u(x[n], y[j]);
	}
	//mu_4
	vector<vector<double>>* cur_matrix = &m_1;
	for (int i = 0; i <= n; i++) {
		if (i > n / 2) {
			cur_matrix = &m_2;
			shift = n / 2 + 1;
		}
		(*cur_matrix)[i - shift][0] = u(x[i], 0);
	}
	//mu_5
	for (int i = 0; i < n/2; i++) {
		m_2[i][m / 2] = u(x[i+n/2+1], y[m / 2]);
	}
	//mu_6
	for (int i = 0; i <= n / 2; i++) {
		m_1[i][m] = u(x[i], y[m]);
	}

}


void show_matrix(const vector<vector<double>>& xy_1,const vector<vector<double>>& xy_2) {
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
void r_s(vector<vector<double>>& r_1, vector<vector<double>>& r_2, vector<vector<double>>& xy_1, vector<vector<double>>& xy_2,vector<double> x,vector<double> y) {
	// Невязка:

	//a_coef = -2*(1/h/h+1/k/k)
	//r^(s) = Ax^(s) - b
	//r^(s)[i][j] = a v[i][j] +1/h/h * v[i-1][j] + 1/h/h * v[i+1][j] + 1/k/k * v[i][j+1] + 1/k/k * v[i][j-1] + f[i][j] 

	size_t n = x.size() - 1;
	size_t m = y.size() - 1;
	double h = x[1] - x[0];
	double k = y[1] - y[0];
	// Невязка:

		//a_coef = -2*(1/h/h+1/k/k)
		//r^(s) = Ax^(s) - b
		//r^(s)[i][j] = a v[i][j] +1/h/h * v[i-1][j] + 1/h/h * v[i+1][j] + 1/k/k * v[i][j+1] + 1/k/k * v[i][j-1] + f[i][j] 
	double a_coef = -2 * (1 / h / h + 1 / k / k);
	
	for (int i = 1; i < n / 2; i++) {
		for (int j = 1; j < m; j++) {
			r_1[i][j] = a_coef * xy_1[i][j] + 1 / h / h * xy_1[i - 1][j] + 1 / h / h * xy_1[i + 1][j] + 1 / k / k * xy_1[i][j + 1] + 1 / k / k * xy_1[i][j - 1] + f(x[i], y[j]);
		}
	}
	for (int i = 1; i < n / 2 - 1; i++) {
		for (int j = 1; j < m / 2; j++) {
			r_2[i][j] = a_coef * xy_2[i][j] + 1 / h / h * xy_2[i - 1][j] + 1 / h / h * xy_2[i + 1][j] + 1 / k / k * xy_2[i][j + 1] + 1 / k / k * xy_2[i][j - 1] + f(x[i + n / 2 + 1], y[j]);
		}
	}
	// невязка на границе двух частей: xy_1 и xy_2
	int i = n / 2;
	for (int j = 1; j < m / 2; j++) {
		r_1[i][j] = a_coef * xy_1[i][j] + 1 / h / h * xy_1[i - 1][j] + 1 / h / h * xy_2[0][j] +
			1 / k / k * xy_1[i][j + 1] + 1 / k / k * xy_1[i][j - 1] +
			f(x[i], y[j]);
	}
	i = 0;
	for (int j = 1; j < m / 2; j++) {
		r_2[i][j] = a_coef * xy_2[i][j] + 1 / h / h * xy_1[n / 2][j] + 1 / h / h * xy_2[i + 1][j] +
			1 / k / k * xy_2[i][j + 1] + 1 / k / k * xy_2[i][j - 1] +
			f(x[n / 2 + 1], y[j]);
	}
}
void step(vector<vector<double>>& xy,const vector<vector<double>>& r_s,double tau) {
	//Численное решение
	// [ x^(s+1) - x^(s) ]/tau +Ax^(s)  = b
	// x^(s+1) = tau*(b - Ax^(s)) + x^(s)  
	// x^(s+1) = x^(s) - tau*(r^(s))
	// Можно оптимизировать(не считать значения на границе)

	for(int i = 0; i<xy.size();i++){
		for (int j = 0; j < xy[0].size(); j++) {
			xy[i][j] = xy[i][j] - tau * r_s[i][j]*(-1);//т.к. матрица не является положительно определенной
		}
	}
}

void simple_iteration_method(int n, int m, int N_max) {
	double a = 0., b = 1., c = 0., d = 1.;
	double h = (b - a) / n, k = (d - c) / m;
	

	//Именно для задачи Дирихле уравнения Пуассона
	double M_max = 4 * (1/h/h+1/k/k);
	double M_min = 0;

	//Временное значение для tau( Нужно оптимизировать его, а именно найти лучшую оценку для собственных чисел)
	/*
	vector<double> x, y;
	for (int i = 0; i <= n; i++) {
		x.push_back(a + h * i);
	}
	for (int i = 0; i <= m; i++) {
		y.push_back(c + k * i);
	}*/

	vector<vector<double>> xy_1(n/2+1, vector<double>(m+1,0)), xy_2(n / 2, vector<double>(m / 2 + 1, 0));
	
	//вспомогательные векторы (значения x, y на сетке)
	vector<double> x, y;
	for (int i = 0; i <= n; i++) {
		x.push_back(a + i * h);
	}
	for (int i = 0; i <= m; i++) {
		y.push_back(c + i * k);
	}
	//Отрисовка частей:
	/*
	std::cout << "Первая часть:\n";
	for (int j = 0; j < xy_1[0].size(); j++){
		for (int i = 0; i < xy_1.size(); i++) {
			std::cout << xy_1[i][j] << ' ';
		}
		std::cout << '\n';
	}
	std::cout << "Вторая часть:\n";
	for (int j = 0; j < xy_2[0].size(); j++) {
		for (int i = 0; i < xy_2.size(); i++) {
			std::cout << xy_2[i][j] << ' ';
		}
		std::cout << '\n';
	}
	*/
	//Заполнение сетки граничными условиями:	
	fill_border_conditions(xy_1,xy_2,x,y,n,m);



	vector<vector<double>>  r_1(n / 2 + 1, vector<double>(m + 1, 0)), r_2(n / 2, vector<double>(m / 2 + 1, 0));

	double tau = 2 / M_max;

	for (int _ = 0; _ < N_max; _++) {


		r_s(r_1, r_2, xy_1, xy_2, x, y);
		step(xy_1,r_1,tau);
		step(xy_2, r_2, tau);
		if (_ % 100 == 0) {
			show_matrix(r_1, r_2);
			cout << "New step:\n\n";
		}
	}

	show_matrix(xy_1, xy_2);
}



int main() {
	system("chcp 1251");
	int n=6, m = 6;
	if (!check_for_multiplicity(n, m)) {
		std::cout << "Сетка не накладывается на область";
		return 0;
	}
	int N_max = 200;
	simple_iteration_method(n, m,N_max);


	return 0;
}