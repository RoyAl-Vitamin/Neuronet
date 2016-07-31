package com.mmsp.neuronet;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;
import javax.imageio.ImageIO;

public class Neuronet {

	//static String[] imageName = {"А.png", "Б.png", "Н.png", "И.png"}; // имена учебных выборок

	static String[] imageName = {"0.png", "1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png", "8.png", "9.png"};

	//static String[] imageName = {"processedS/вс_1.png", "processedS/вс_2.png", "processedS/вс_3.png", "processedS/вс_4.png", "processedS/дс_1.png", "processedS/дс_2.png", "processedS/пп_1.png", "processedS/пп_2.png", "processedS/пп_3.png", "processedS/пп_4.png"};

	static double liX[][] = new double[imageName.length][]; // массив массива разложенных по цвету картинок (обучающих выборок)

	static double W[][]; // Обучающая матрица синоптических весов

	static double Y[]; // вектор с шумами, который будем распознавать

	static int n; // количество нейронов = количеству элементов в строке иил столбце матрицы W

	static double EPS = 1E-4; // Точность

	static double ny = 0.75; // Скорость обучения - число из [0.7; 0.9]

	static int tye = 0; // Глобальный счётчик

	/**
	 * выбор метода обучения задаёт переменная rt
	 * переключать и задавать обучающие выборки вручную
	 * @param args
	 */
	public static void main(String[] args) {

		loadImg(); // Загрузка исходных образов для обучения

		for (int ew = 0; ew < 10; ew++) {

			Y = noise("2.png", 0.9); // 0.9 => 90 %
	
			//save(Y);
			//Y = load("processedS/вс_1.png"); // Загрузка зашумлённого образа
			//Y = load("_Н.png"); // Загрузка зашумлённого образа
	
			initW(); // инициализация матрицы W (задание размеров и заполнение её 0-ми)
	
			int rt = 1; // переменная выбора метода обучения сети
	
			double result[] = null;
			switch (rt) {
			case 0:
				result = byHebb(); // по правилу обучения Хебба
				break;
			case 1:
				result = byProjection(); // обучение по методу проекций 
				break;
			case 2:
				result = byDeltaProjection(); // обучение по методу Delta-проекций
				break;
			default:
				result = byStandart(); // по стандартному правилу обучения W = Summ[ X_i^T X_i ]
			}
			analyze(result);
		}
	}

	/**
	 * Сравнивает по норме результат работы нейросети с обучающей выборкой
	 * @param result результат работы нейросети
	 */
	private static void analyze(double[] result) {
		System.out.println("Эксперимент №" + tye++);
		double diff[] = new double[n];
		for (int t = 0; t < liX.length; t++) {
			for (int i = 0; i < n; i++) {
				diff[i] = Math.abs(liX[t][i] - result[i]);
			}
			double norm  = norm(diff);
			System.out.println("norm(" + imageName[t] + " - result) == " + norm + " количество отличающихся пикслей == " + Math.round(Math.pow(norm / 2, 2)));
		}
	}

	/**
	 * сделаем зашумление исходному образу
	 * @param sValue имя исходного образа
	 * @param p вероятность того, что цвет останется тем же
	 * @return вектор зашумлённого изображения
	 */
	private static double[] noise(String sValue, double p) {

		if (p > 1) p = 1;
		if (p < 0) p = 0;
		double liTemp[] = load(sValue); // загрузим изначальное изображение

		Random r = new Random();
		/* Будем брать рандомное число r in [0;1) и сравнивать его с верояностью p смены цвета */
		for (int i = 0; i < liTemp.length; i++) {// зашумим
			if (r.nextDouble() > p) // Если больше, то поменяем текущую компоненту на другой цвет
				liTemp[i] *= -1;
			// Иначе, пусть она останется такого же цвета
		}
		save(liTemp, "noise_" + tye + "_" + sValue);
		return liTemp;
	}

	private static double[] byStandart() {

		System.out.println("Метод обучения по стандарту W = Summ[ X_i^T X_i ]");
		System.out.println("Максимально количество образов, которые может запомнить нейронная сеть == " + (int)(n * Math.log(2) / (2 * Math.log(n))));

		learnStandart();

		double[] liY_new = new double[n]; // на новом шаге
		double[] liY_old = new double[n]; // на предыдущем шаге
		
		for (int i = 0; i < Y.length; i++) liY_new[i] = liY_old[i] = Y[i]; // скопируем вектор Y

		/*for (int i = 0; i < imageName.length; i++) // Выведем ветора, посмотрим чё там
			writeVector(liX.get(i));
		writeVector(Y);*/

		double minNorm = EPS + 1;
		int k = 0;

		while (true) {

			for (int i = 0; i < liY_old.length; i++) liY_old[i] = liY_new[i]; // перекопируем в старый вектор из нового
			// перемножаем W и Y
			for (int i = 0; i < n; i++) { // бежим по строкам матрицы W
				int summ = 0;
				for (int j = 0; j < n; j++) { // по компонентам вектора Y
					summ += W[i][j] * liY_old[j];
				}
				// Y = sign(Y)
				if (summ >= 0) liY_new[i] = 1.0; else liY_new[i] = -1.0; // ступенчатая функция
			}

			boolean b = false;

			double liRES[] = new double[n];
			for (int i = 0; i < n; i++) liRES[i] = liY_new[i] - liY_old[i];
			if (norm(liRES) < EPS) {
				System.err.println("Свалился в ложный аттрактор norm == " + norm(liRES)); // Проверка неравенста векторов
				b = true;
			}

			//save(liY_new);
			if (!b)
				for (int i = 0; i < liX.length; i++) { // условие выхода из цикла
	
					double liDiff[] = new double[n]; // вектор разности _Y and _Y*
					for (int j = 0; j < n; j++) liDiff[j] = liY_new[j] - liX[i][j];
	
					double currNorm = norm(liDiff); // Подсчёт нормы от разности 2-ух векторов
	
					if (currNorm < minNorm) minNorm = currNorm;
					if (currNorm <= EPS) { // Условие выхода из цикла
						System.err.println("Step == " + k +" Current Norm == " + currNorm);
						b = true;
						break;
					}
				}
			if (b) break; // выход из внешнего цикла
			k++;
		}
		save(liY_new, null);
		return liY_new;
	}

	private static double[] byDeltaProjection() {

		System.out.println("Метод обучения Дельта проекций");

		learnDeltaProjection();

		double liY_new[] = new double[n]; // на новом шаге
		double liY_old[] = new double[n]; // на предыдущем шаге
		for (int i = 0; i < Y.length; i++) liY_new[i] = liY_old[i] = Y[i]; // скопируем вектор Y

		/*for (int i = 0; i < imageName.length; i++) // Выведем ветора, посмотрим чё там
			writeVector(liX[i]);
		writeVector(Y);*/

		double minNorm = EPS + 1;
		int k = 0;

		while (true) {

			for (int i = 0; i < n; i++) liY_old[i] = liY_new[i]; // перекопируем в старый вектор из "нового"

			// перемножаем W и Y
			for (int i = 0; i < n; i++) { // бежим по строкам матрицы W
				double summ = 0;
				for (int j = 0; j < n; j++) { // по компонентам вектора Y
					summ += W[i][j] * liY_new[j];
				}
				// Y = sign(Y)
				if (summ >= 0) liY_new[i] = 1.0; else liY_new[i] = -1.0;
			}

			boolean b = false;

			double liRES[] = new double[n];
			for (int i = 0; i < n; i++) liRES[i] = liY_new[i] - liY_old[i];
			if (norm(liRES) < EPS) {
				System.err.println("Свалился в ложный аттрактор norm(Y_old - Y_new) == " + norm(liRES) + " на шаге k == " + k); // Проверка неравенста векторов
				b = true;
			}
			//save(liY_new, null);

			if (!b)
				for (int i = 0; i < liX.length; i++) { // условие выхода из цикла
	
					double liDiff[] = new double[n]; // вектор разности _Y and _Y*
					for (int j = 0; j < n; j++) liDiff[j] = liY_new[j] - liX[i][j];
	
					double currNorm = norm(liDiff); // Подсчёт нормы от разности 2-ух векторов
	
					if (currNorm < minNorm) minNorm = currNorm;
					if (currNorm <= EPS) { // Условие выхода из цикла
						System.err.println("Step == " + k +" Current Norm == " + currNorm);
						b = true;
						break;
					}
				}
			if (b) break; // выход из внешнего цикла
			k++;
		}
		save(liY_new, null);
		return liY_new;
	}

	private static double[] byProjection() {

		System.out.println("Метод обучения проекциями, ёмкость сети == " + (n - 1));

		learnProjection();

		double liY_new[] = new double[n]; // на новом шаге
		double liY_old[] = new double[n]; // на предыдущем шаге
		for (int i = 0; i < Y.length; i++) liY_new[i] = liY_old[i] = Y[i]; // скопируем вектор Y

		/*for (int i = 0; i < imageName.length; i++) // Выведем ветора, посмотрим чё там
			writeVector(liX[i]);
		writeVector(Y);*/

		double minNorm = EPS + 1;
		int k = 0;

		while (true) {

			for (int i = 0; i < n; i++) liY_old[i] = liY_new[i]; // перекопируем в старый вектор из "нового"

			// перемножаем W и Y
			for (int i = 0; i < n; i++) { // бежим по строкам матрицы W
				double summ = 0;
				for (int j = 0; j < n; j++) { // по компонентам вектора Y
					summ += W[i][j] * liY_new[j];
				}
				// Y = sign(Y)
				if (summ >= 0) liY_new[i] = 1.0; else liY_new[i] = -1.0;
			}

			boolean b = false;

			double liRES[] = new double[n];
			for (int i = 0; i < n; i++) liRES[i] = liY_new[i] - liY_old[i];
			if (norm(liRES) < EPS) {
				System.err.println("Свалился в ложный аттрактор norm(Y_old - Y_new) == " + norm(liRES) + " на шаге k == " + k); // Проверка неравенста векторов
				b = true;
			}
			//save(liY_new, null);

			if (!b)
				for (int i = 0; i < liX.length; i++) { // условие выхода из цикла
	
					double liDiff[] = new double[n]; // вектор разности _Y and _Y*
					for (int j = 0; j < n; j++) liDiff[j] = liY_new[j] - liX[i][j];
	
					double currNorm = norm(liDiff); // Подсчёт нормы от разности 2-ух векторов
	
					if (currNorm < minNorm) minNorm = currNorm;
					if (currNorm <= EPS) { // Условие выхода из цикла
						System.err.println("Step == " + k +" Current Norm == " + currNorm);
						b = true;
						break;
					}
				}
			if (b) break; // выход из внешнего цикла
			k++;
		}
		save(liY_new, null);
		return liY_new;
	}


	private static double[] byHebb() {

		System.out.println("Метод обучения по правилу Хебба");
		System.out.println("При Eps = 0.01 ёмкость сети == " + (int)(0.138 * n));

		learnHebb();

		double liY_new[] = new double[n]; // на новом шаге
		double liY_old[] = new double[n]; // на предыдущем шаге
		for (int i = 0; i < Y.length; i++) liY_new[i] = liY_old[i] = Y[i]; // скопируем вектор Y

		/*for (int i = 0; i < imageName.length; i++) // Выведем ветора, посмотрим чё там
			writeVector(liX[i]);
		writeVector(Y);*/

		double minNorm = EPS + 1;
		int k = 0;

		double W_old[][] = new double[n][n];

		while (true) {

			for (int i = 0; i < liY_old.length; i++) liY_old[i] = liY_new[i]; // перекопируем в старый вектор из "нового"

			// перекопируем значение матрицы W_old
			for (int i = 0; i < W.length; i++)
				for (int j = 0; j < W[i].length; j++)
					W_old[i][j] = W[i][j];

			// перемножаем W и Y
			for (int i = 0; i < n; i++) { // бежим по строкам матрицы W
				double summ = 0;
				for (int j = 0; j < n; j++) { // по компонентам вектора Y
					summ += W[i][j] * liY_new[j];
				}
				// Y = sign(Y)
				if (summ >= 0) liY_new[i] = 1.0; else liY_new[i] = -1.0;
			}

			/* переобучение по гипотезе Хебба*/
			// W(i+1) = W(i) + dW
			for (int i = 0; i < W.length; i++)
				for (int j = 0; j < W[i].length; j++) {
					W[i][j] = W[i][j] + ny * liY_new[i] * liY_old[j];
				}

			/* переобучение по гипотезе ковариации*/
			// W(i+1) = W(i) + dW
			/*for (int i = 0; i < W.length; i++)
				for (int j = 0; j < W[i].length; j++) {
					W[i][j] = W[i][j] + ny * (liY_new[i] - avg(liY_new)) * (liY_old[j] - avg(liY_old));
				}*/


			// Посмотрим разницу после переобучения
			double W_temp[][] = new double[n][n];

			for (int i = 0; i < W.length; i++)
				for (int j = 0; j < W[i].length; j++)
					W_temp[i][j] = Math.abs(W[i][j] - W_old[i][j]);
			System.out.println("Норма разности матриц W_i и W_i-1 == " + normMatrix(W_temp)); // без обучения даёт 0, с обучением даёт около 75, в чём подвох?

			boolean b = false;

			double liRES[] = new double[n];
			for (int i = 0; i < n; i++) liRES[i] = liY_new[i] - liY_old[i];
			if (norm(liRES) < EPS) {
				System.err.println("Свалился в ложный аттрактор norm(Y_old - Y_new) == " + norm(liRES)); // Проверка неравенста векторов
				b = true;
			}
			//save(liY_new, null);

			if (!b)
				for (int i = 0; i < liX.length; i++) { // условие выхода из цикла
	
					double liDiff[] = new double[n]; // вектор разности _Y and _Y*
					for (int j = 0; j < n; j++) liDiff[j] = liY_new[j] - liX[i][j];
	
					double currNorm = norm(liDiff); // Подсчёт нормы от разности 2-ух векторов
	
					if (currNorm < minNorm) minNorm = currNorm;
					if (currNorm <= EPS) { // Условие выхода из цикла
						System.out.println("Step == " + k +" Current Norm == " + currNorm);
						b = true;
						break;
					}
				}
			if (b) break; // выход из внешнего цикла
			k++;
		}
		save(liY_new, null);
		return liY_new;
	}

	/**
	 * Считает норму матрицы
	 * @param w_temp матрицы
	 * @return её норма
	 */
	private static double normMatrix(double[][] w_temp) {
		double summ = 0;
		for (int i = 0; i < w_temp.length; i++)
			for (int j = 0; j < w_temp[i].length; j++) {
				summ += Math.pow(w_temp[i][j], 2);
			}
		return Math.sqrt(summ);
	}

	/**
	 * считает средние от вектора
	 * @param liY_new входной вектор
	 * @return его среднее
	 */
	private static double avg(double[] liY_new) {
		double summ = 0;
		for (int i = 0; i < liY_new.length; i++)
			summ += liY_new[i];
		return summ / liY_new.length;
	}

	/**
	 * Евклидова норма Sqrt[ Summ[i*i, {i, 0, n*n}] ]
	 * @param diff входной вектор
	 * @return его норма
	 */
	private static double norm(double[] diff) {
		double summ = 0;
		for (int i = 0; i < diff.length; i++) summ += Math.pow(diff[i], 2);
		return Math.sqrt(summ);
	}

	/**
	 * Обучение методом Дельта-проекций
	 */
	private static void learnDeltaProjection() {

		double W_old[][] = new double[n][n]; // инициализируем матрицу W_old для хранения предыдущего шага
		double W_new[][] = new double[n][n]; // инициализируем матрицу W_new для хранения текущего шага
		double W_temp[][] = new double[n][n]; // инициализируем матрицу W_temp для хранения W_new - W_old для посчёдта нормы W_new - W_old

		double liTemp[] = new double[n]; // Просто вспомогательный вектор для расчётов

		while (true) {
			for (int t = 0; t < imageName.length; t++) { // предявим каждый обучающий образ
				for (int i = 0; i < n; i++)
					for (int j = 0; j < n; j++)
						W_old[i][j] = W_new[i][j]; // копирование матрицы с предыдущего шага

				/* Подсчёт выражения W_i-1 * x_i */
				for (int i = 0; i < n; i++) {
					double summ = 0;
					for (int j = 0; j < n; j++)
						summ += W_old[i][j] * liX[t][j];
					liTemp[i] = summ;
				}

				/* Подсчёт x_i - W_i-1 * x_i */
				for (int i = 0; i < n; i++)
					liTemp[i] = liX[t][i] - liTemp[i];

				/* Подсчёт W_i-1 + (x_i - W_i-1 * x_i) * x_i^T * ny / n и запишем это в W_new */
				for (int i = 0; i < n; i++)
					for (int j = 0; j < n; j++)
						W_new[i][j] = W_old[i][j] + ny * (liTemp[i] * liX[t][j]) / n;
			}

			for (int i = 0; i < n; i++) // составим матрицу W_new - W_old для подсчёта нормы
				for (int j = 0; j < n; j++)
					W_temp[i][j] = W_new[i][j] - W_old[i][j];
			if (normMatrix(W_temp) <= EPS) break; // условие выхода - до стабилизации значения
		}

		// перегоним всё это в W
		for (int i = 0; i < W_old.length; i++)
			for (int j = 0; j < W_old[i].length; j++)
				W[i][j] = W_new[i][j];
		System.out.println("Обучение закончнео, матрица весов стабилизировалась");
	}

	/**
	 * Обучение методом проекций
	 */
	private static void learnProjection() {

		double W_old[][] = new double[n][n]; // инициализируем матрицу W_old для хранения предыдущего шага
		double W_new[][] = new double[n][n]; // инициализируем матрицу W_new для хранения текущего шага


		/* неудачно!
		 * попытка посчитать:
		 * y_i = (W_i-1 - E) x_i
		 * W_i = W_i-1 - (y_i * y_i^T) / (y_i^T * y_i)
		 */
		/*List<Double> liY = new ArrayList<>(n); // инициализируем вектор для хранения y_i = (W_i-1 - E) * x_i, x_i - i-ая обучающая выборка 
		for (int j = 0; j < n; j++)
			liY.add(0.0);

		for (int t = 0; t < imageName.length; t++) {

			for (int i = 0; i < W_old.size(); i++)
				for (int j = 0; j < W_old.get(i).size(); j++)
					W_old.get(i).set(j, W_new.get(i).get(j)); // копирование матрицы с предыдущего шага

			for (int i = 0; i < W_old.size(); i++) { // подготовка вектора y_i
				double summ = 0;
				for (int j = 0; j < W_old.get(i).size(); j++) {
					if (i == j)	summ += (W_old.get(i).get(j) - 1) * liX.get(t).get(i); // вычитая из матрицы единичную матрицу
					else summ += W_old.get(i).get(j) * liX.get(t).get(i);
				}
				liY.set(i, summ);
			}

			// далее идёт подсчёт выражения W_i = W_i-1 - ((y_i * y_i^T) / (y_i^T * y_i))
			double dISP = internalScalarProduct(liY); // подсчёт выражения вида y_i^T * y_i

			for (int i = 0; i < W_old.size(); i++)
				for (int j = 0; j < W_old.get(i).size(); j++)
					W_new.get(i).set(j, W_old.get(i).get(j) - (liY.get(i) * liY.get(j)) / dISP);
		}*/

		/*
		 * попытка посчитать:
		 * W_i = W_i-1 + (W_i-1 * x_i - x_i) * (W_i-1 * x_i - x_i)^T / (x_i^T * x_i - x_i^T * W_i-1 * x_i)
		 */
		for (int t = 0; t < imageName.length; t++) {

			for (int i = 0; i < W_old.length; i++)
				for (int j = 0; j < W_old[i].length; j++)
					W_old[i][j] = W_new[i][j]; // копирование матрицы с предыдущего шага

			/* посчитаем знаменатель x_i^T * x_i - x_i^T * W_i-1 * x_i */
			double denominator = internalScalarProduct(liX[t]); // выдаст: x_i^T * x_i
			for (int i = 0; i < n; i++) {
				double summ = 0;
				for (int j = 0; j < n; j++) { // подсчёт i-ой компоненты вектора x_i^T * W_i-1
					summ += liX[t][j] * W_old[j][i];
				}
				denominator -= liX[t][i] * summ; // покомпонентный вычет из x_i^T * x_i компоненты вектора x_i^T * W_i-1 умноженой на соответствующую компоненту вектора x_i
			}

			/* теперь посчитаем вектор W_i-1 * x_i - x_i */
			double liTemp[] = new double[n];
			for (int i = 0; i < n; i++) {
				double temp = 0;
				for (int j = 0; j < n; j++)
					temp += W_old[i][j] * liX[t][j];
				liTemp[i] = temp;
			} // только что посчитано выражение W_i-1 * x_i
			// теперь вычтем из W_i-1 * x_i вектор x_i
			for (int i = 0; i < n; i++)
				liTemp[i] = liTemp[i] - liX[t][i];
			// получили вектор W_i-1 * x_i - x_i

			/* посчитаем матрицу W_i покомпонентно, т.е. W_i[ij] = W_i-1[ij] + (W_i-1 * x_i - x_i)[i] * (W_i-1 * x_i - x_i)[j] / (x_i^T * x_i - x_i^T * W_i-1 * x_i) */
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					W_new[i][j] = W_old[i][j] + liTemp[i] * liTemp[j] / denominator;
				}
			}
		}

		// перегоним всё это в W
		for (int i = 0; i < W_old.length; i++)
			for (int j = 0; j < W_old[i].length; j++)
				W[i][j] = W_new[i][j];
	}

	/**
	 * Внутреннее скалярное произведение: y_i^T * y_i
	 * @param liY вектор-столбец
	 * @return результат скалярного произведения
	 */
	private static double internalScalarProduct(double liY[]) {
		double summ = 0;
		for (int i = 0; i < liY.length; i++) {
			summ += Math.pow(liY[i], 2);
		}
		return summ;
	}

	/**
	 * Обучение методом Хебба
	 */
	private static void learnHebb() { // Рандомная матрица [-1, 0, 1], а на диагонали 0
		// TODO Уточнить изначальный вид матрицы
		/*Random r = new Random();
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++)
				W.get(i).set(j, (double) (r.nextInt(3) - 1));
			W.get(i).set(i, 0.0);
		}*/

		//learnStandart();

		for (int t = 0; t < imageName.length; t++)
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++)
					W[i][j] = W[i][j] + (liX[t][i] * liX[t][j]) / n;
				W[i][i] = 0.0;
			}
	}

	/**
	 * Обобщённое правило Хебба
	 */
	private static void learnStandart() { // W = Summ(x_k^t * x_k, {k, 1, imageName.length})
		for (int i = 0; i < imageName.length; i++) {
			add(W, liX[i]);
		}
	}

	/**
	 * Перемножает вектор x_k сам на себя, создавая матрицу (x_k^T * x_k)
	 * Чистит диагональ
	 * Складывает с матрицей W (w2)
	 * @param w2 матрица W
	 * @param liX2 вектор x_k из обучающей выборки
	 */
	private static void add(double[][] w2, double[] liX2) {
		// перемножим : x_k^T * x_k и сразу сложим с w2 и почистим диагональ на каждом шаге, чтоб быстрее было
		// в результате получим симметричную матрицу A^T = A
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++)
				w2[i][j] = w2[i][j] + liX2[i] * liX2[j];
			w2[i][i] = 0.0; // обнуляем диагональ
		}
	}

	/**
	 * обнулим и зададим размер матрице W
	 */
	private static void initW() {
		W = new double[n][n]; // заполняет матрицу 0.0
	}

	/**
	 * Вывод матрицы matrix в удобном для перегона в Вольфрам виде (для проверки)
	 */
	private static void writeMatrix(double matrix[][]) {
		System.out.print("{");
		for (int i = 0; i < n; i++) {
			System.out.print("{" + matrix[i][0]);
			for (int j = 1; j < n; j++)
				System.out.print("," + matrix[i][j]);
			if (i != n - 1)
				System.out.println("},");
			else
				System.out.print("}");
		}
		System.out.println("}");
	}

	/**
	 * Вывод вектора на экран в виде [-1, 1] в удобном для перегона в Вольфрам виде (для проверки)
	 * @param liX2 
	 */
	private static void writeVector(double[] liX2) {
		System.out.print("{" + liX2[0]);
		for (int i = 1; i < liX2.length; i++) {
			System.out.print(","+ liX2[i]);
		}
		System.out.println("}");
	}

	/**
	 * Переведём картинки обучающих выборок в массивы [-1, 1], запишем их в liX
	 */
	private static void loadImg() {
		for (int i = 0; i < imageName.length; i++) { // пройдёмся по всем именам картинок
			liX[i] = load(imageName[i]);
		}
	}

	/**
	 * Загрузка каритнки, усредняем цвет для перевода его в оттенки серого
	 * @param sValue имя картинки, лежащей в ".\\src\\main\\resources\\"
	 * @return
	 */
	private static double[] load(String sValue) {
		BufferedImage img = null;
		try {
			img = ImageIO.read(new File(".\\src\\main\\resources\\" + sValue));
			n = img.getHeight() * img.getWidth();
			//if (liX != null) liX = new double[][];
		} catch (IOException e) {
			System.err.println("Не удалось найти файл " + sValue);
		}

		// перегоним картинки в массивы {-1, 1} // -1 Black : 1 White
		double arTemp[] = new double[n];
		int k = 0;
		for (int x = 0; x < img.getHeight(); x++)
			for (int y = 0; y < img.getWidth(); y++) {
				int clr = img.getRGB(y, x); // достаём цвет пикселя
				int  red = (clr & 0x00ff0000) >> 16; // смотрим содержание
				int  green = (clr & 0x0000ff00) >> 8;
				int  blue  =  clr & 0x000000ff;
				int avg = (red + green + blue) / 3;
				arTemp[k++] = avg * 2.0 / 255 - 1;
			}
		return arTemp;
	}

	/**
	 * Вывод вектора в изображение
	 * @param liTemp входящий вектор
	 * @param name предпочтительное имя для сохранения
	 */
	private static void save(double[] liTemp, String name) {
		BufferedImage img = new BufferedImage((int) Math.sqrt(n), (int) Math.sqrt(n), BufferedImage.TYPE_INT_RGB);
		int k = 0;
		int rgb;
		for (int y = 0; y < Math.sqrt(n); y++)
			for (int x = 0; x < Math.sqrt(n); x++) {
				int temp = (int) ((liTemp[k] + 1) * 255 / 2); // перегон из [-1; 1] в [0; 255]
				rgb = (new Color(temp, temp, temp)).getRGB(); // 255 оттенков серого
				img.setRGB(x, y, rgb);
				k++;
			}
		try {
			File outputfile = null;
			if (name == null)
				outputfile = new File("outputImg_Step_" + tye + ".png");
			else
				outputfile = new File(name);
		    ImageIO.write(img, "png", outputfile);
		} catch (IOException e) { // когда чёт не пошло
			System.err.println("Так получилось в общем, что, кажись, файл не захотел создаваться");
		}
	}
}
