package com.mmsp.neuronet;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

public class Neuronet {

	//static String[] imageName = {"А.png", "Б.png", "Н.png", "И.png"}; // имена учебных выборок

	static String[] imageName = {"0.png", "1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png", "8.png", "9.png"};

	static List<ArrayList<Double>> liX = new ArrayList<ArrayList<Double>>(); // массив массива разложенных по цвету картинок (обучающих выборок)

	static List<ArrayList<Double>> W = new ArrayList<ArrayList<Double>>(); // Обучающая матрица синоптических весов

	static List<Double> Y = null; // вектор с шумами, который будем распознавать

	static int n; // количество нейронов = количеству элементов в строке иил столбце матрицы W

	static double EPS = 1E-4; // Точность

	static double ny = 0.75; // Скорость обучения

	static int tye = 0; // Глобальный счётчик

	/**
	 * выбор метода обучения задаёт переменная rt
	 * переключать и задавать обучающие выборки вручную
	 * @param args
	 */
	public static void main(String[] args) {

		loadImg(); // Загрузка исходных образов для обучения

		//Y = noise("_А.png", 0.8); // 0.9 => 90 %

		//save(Y);
		Y = load("_4.png"); // Загрузка зашумлённого образа
		//Y = load("_Н.png"); // Загрузка зашумлённого образа

		initW(); // инициализация матрицы W (задание размеров и заполнение её 0-ми)

		int rt = 2; // переменная выбора метода обучения сети

		switch (rt) {
		case 0:
			byHebb(); // по правилу обучения Хебба
			break;
		case 1:
			byProjection(); // обучение по методу проекций 
			break;
		case 2:
			byDeltaProjection(); // обучение по методу Delta-проекций
			break;
		default:
			byStandart(); // по стандартному правилу обучения W = Summ[ X_i^T X_i ]
		}
	}

	/**
	 * сделаем зашумление исходному образу
	 * @param sValue имя исходного образа
	 * @param x доля зашмления
	 * @return вектор зашумлённого изображения
	 */
	private static List<Double> noise(String sValue, double x) {
		// UNDONE пока просто делает изображение более блеклым, подумать над M * X
		if (x > 1) x = 1;
		if (x < 0) x = 0;
		List<Double> liTemp = load(sValue); // загрузим изначальное изображение

		/* будем перемножать матрицу M, у которой на диагонали будет x * length[-1;1] / length[0;1] - 1, на вектор исходного изображения liTemp
		 * фактически liTemp[i] * (2 * x - 1)
		 */
		for (int i = 0; i < liTemp.size(); i++) // зашумим
			liTemp.set(i, liTemp.get(i) * (2 * x - 1));
		return liTemp;
	}

	private static void byStandart() {

		System.out.println("Метод обучения по стандарту W = Summ[ X_i^T X_i ]");
		System.out.println("Максимально количество образов, которые может запомнить нейронная сеть == " + (int)(n * Math.log(2) / (2 * Math.log(n))));

		learnStandart();

		List<Double> liY_new = new ArrayList<>(n); // на новом шаге
		List<Double> liY_old = new ArrayList<>(n); // на предыдущем шаге
		for (Double i : Y) {liY_new.add(i); liY_old.add(i);} // скопируем вектор Y

		/*for (int i = 0; i < imageName.length; i++) // Выведем ветора, посмотрим чё там
			writeVector(liX.get(i));
		writeVector(Y);*/

		double minNorm = EPS + 1;
		int k = 0;

		while (true) {

			for (int i = 0; i < liY_old.size(); i++) liY_old.set(i, liY_new.get(i)); // перекопируем в старый вектор из нового
			// перемножаем W и Y
			for (int i = 0; i < n; i++) { // бежим по строкам матрицы W
				int summ = 0;
				for (int j = 0; j < n; j++) { // по компонентам вектора Y
					summ += W.get(i).get(j) * liY_old.get(j);
				}
				// Y = sign(Y)
				if (summ >= 0) liY_new.set(i, 1.0); else liY_new.set(i, -1.0); // ступенчатая функция
			}

			boolean b = false;

			List<Double> liRES = new ArrayList<>(n);
			for (int i = 0; i < n; i++) liRES.add(liY_new.get(i) - liY_old.get(i));
			if (norm(liRES) < EPS) {
				System.err.println("Свалился в ложный аттрактор norm == " + norm(liRES)); // Проверка неравенста векторов
				b = true;
			}

			//save(liY_new);
			if (!b)
				for (int i = 0; i < liX.size(); i++) { // условие выхода из цикла
	
					List<Double> liDiff = new ArrayList<>(n); // вектор разности _Y and _Y*
					for (int j = 0; j < n; j++) liDiff.add(liY_new.get(j) - liX.get(i).get(j));
	
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
		save(liY_new);
		
	}

	private static void byDeltaProjection() {

		System.out.println("Метод обучения Дельта проекций");

		learnDeltaProjection();

		List<Double> liY_new = new ArrayList<>(n); // на новом шаге
		List<Double> liY_old = new ArrayList<>(n); // на предыдущем шаге
		for (Double i : Y) {liY_new.add(i); liY_old.add(i);} // скопируем вектор Y

		/*for (int i = 0; i < imageName.length; i++) // Выведем ветора, посмотрим чё там
			writeVector(liX.get(i));
		writeVector(Y);*/

		double minNorm = EPS + 1;
		int k = 0;

		while (true) {

			for (int i = 0; i < n; i++) liY_old.set(i, liY_new.get(i)); // перекопируем в старый вектор из "нового"

			// перемножаем W и Y
			for (int i = 0; i < n; i++) { // бежим по строкам матрицы W
				double summ = 0;
				for (int j = 0; j < n; j++) { // по компонентам вектора Y
					summ += W.get(i).get(j) * liY_new.get(j);
				}
				// Y = sign(Y)
				if (summ >= 0) liY_new.set(i, 1.0); else liY_new.set(i, -1.0);
			}

			boolean b = false;

			List<Double> liRES = new ArrayList<>(n);
			for (int i = 0; i < n; i++) liRES.add(liY_new.get(i) - liY_old.get(i));
			if (norm(liRES) < EPS) {
				System.err.println("Свалился в ложный аттрактор norm(Y_old - Y_new) == " + norm(liRES) + " на шаге k == " + k); // Проверка неравенста векторов
				b = true;
			}
			//save(liY_new);

			if (!b)
				for (int i = 0; i < liX.size(); i++) { // условие выхода из цикла
	
					List<Double> liDiff = new ArrayList<>(n); // вектор разности _Y and _Y*
					for (int j = 0; j < n; j++) liDiff.add(liY_new.get(j) - liX.get(i).get(j));
	
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
		save(liY_new);
	}

	private static void byProjection() {

		System.out.println("Метод обучения проекциями, ёмкость сети == " + (n - 1));

		learnProjection();

		List<Double> liY_new = new ArrayList<>(n); // на новом шаге
		List<Double> liY_old = new ArrayList<>(n); // на предыдущем шаге
		for (Double i : Y) {liY_new.add(i); liY_old.add(i);} // скопируем вектор Y

		/*for (int i = 0; i < imageName.length; i++) // Выведем ветора, посмотрим чё там
			writeVector(liX.get(i));
		writeVector(Y);*/

		double minNorm = EPS + 1;
		int k = 0;

		while (true) {

			for (int i = 0; i < n; i++) liY_old.set(i, liY_new.get(i)); // перекопируем в старый вектор из "нового"

			// перемножаем W и Y
			for (int i = 0; i < n; i++) { // бежим по строкам матрицы W
				double summ = 0;
				for (int j = 0; j < n; j++) { // по компонентам вектора Y
					summ += W.get(i).get(j) * liY_new.get(j);
				}
				// Y = sign(Y)
				if (summ >= 0) liY_new.set(i, 1.0); else liY_new.set(i, -1.0);
			}

			boolean b = false;

			List<Double> liRES = new ArrayList<>(n);
			for (int i = 0; i < n; i++) liRES.add(liY_new.get(i) - liY_old.get(i));
			if (norm(liRES) < EPS) {
				System.err.println("Свалился в ложный аттрактор norm(Y_old - Y_new) == " + norm(liRES) + " на шаге k == " + k); // Проверка неравенста векторов
				b = true;
			}
			//save(liY_new);

			if (!b)
				for (int i = 0; i < liX.size(); i++) { // условие выхода из цикла
	
					List<Double> liDiff = new ArrayList<>(n); // вектор разности _Y and _Y*
					for (int j = 0; j < n; j++) liDiff.add(liY_new.get(j) - liX.get(i).get(j));
	
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
		save(liY_new);
	}


	private static void byHebb() {

		System.out.println("Метод обучения по правилу Хебба");
		System.out.println("При Eps = 0.01 ёмкость сети == " + (int)(0.138 * n));

		learnHebb();

		List<Double> liY_new = new ArrayList<>(n); // на новом шаге
		List<Double> liY_old = new ArrayList<>(n); // на предыдущем шаге
		for (Double i : Y) {liY_new.add(i); liY_old.add(i);} // скопируем вектор Y

		/*for (int i = 0; i < imageName.length; i++) // Выведем ветора, посмотрим чё там
			writeVector(liX.get(i));
		writeVector(Y);*/

		double minNorm = EPS + 1;
		int k = 0;

		/**/
		List<ArrayList<Double>> W_old = new ArrayList<ArrayList<Double>>();
		for (int i = 0; i < n; i++) {
			List<Double> liRow = new ArrayList<>(n);
			for (int j = 0; j < n; j++)
				liRow.add(0.0);
			W_old.add((ArrayList<Double>) liRow);
		}
		/**/

		while (true) {

			for (int i = 0; i < liY_old.size(); i++) liY_old.set(i, liY_new.get(i)); // перекопируем в старый вектор из "нового"

			/**/
			// перекопируем значение матрицы W_old
			for (int i = 0; i < W.size(); i++)
				for (int j = 0; j < W.get(i).size(); j++)
					W_old.get(i).set(j, W.get(i).get(j));
			/**/
			// перемножаем W и Y
			for (int i = 0; i < n; i++) { // бежим по строкам матрицы W
				double summ = 0;
				for (int j = 0; j < n; j++) { // по компонентам вектора Y
					summ += W.get(i).get(j) * liY_new.get(j);
				}
				// Y = sign(Y)
				if (summ >= 0) liY_new.set(i, 1.0); else liY_new.set(i, -1.0);
			}

			/* переобучение по гипотезе Хебба*/
			// W(i+1) = W(i) + dW
			for (int i = 0; i < W.size(); i++)
				for (int j = 0; j < W.get(i).size(); j++) {
					W.get(i).set(j, W.get(i).get(j) + ny * liY_new.get(i) * liY_old.get(j));
				}

			/* переобучение по гипотезе ковариации*/
			// W(i+1) = W(i) + dW
			/*for (int i = 0; i < W.size(); i++)
				for (int j = 0; j < W.get(i).size(); j++) {
					W.get(i).set(j, W.get(i).get(j) + ny * (liY_new.get(i) - avg(liY_new)) * (liY_old.get(j) - avg(liY_old)));
				}*/

			/**/
			// Посмотрим разницу после переобучения
			List<ArrayList<Double>> W_temp = new ArrayList<ArrayList<Double>>();
			for (int i = 0; i < n; i++) {
				List<Double> liRow = new ArrayList<>(n);
				for (int j = 0; j < n; j++)
					liRow.add(0.0);
				W_temp.add((ArrayList<Double>) liRow);
			}
			for (int i = 0; i < W.size(); i++)
				for (int j = 0; j < W.get(i).size(); j++)
					W_temp.get(i).set(j, Math.abs(W.get(i).get(j) - W_old.get(i).get(j)));
			System.out.println("Норма разности матриц W_i и W_i-1 == " + normMatrix(W_temp)); // без обучения даёт 0, с обучением даёт около 75, в чём подвох?
			/**/

			boolean b = false;

			List<Double> liRES = new ArrayList<>(n);
			for (int i = 0; i < n; i++) liRES.add(liY_new.get(i) - liY_old.get(i));
			if (norm(liRES) < EPS) {
				System.err.println("Свалился в ложный аттрактор norm(Y_old - Y_new) == " + norm(liRES)); // Проверка неравенста векторов
				b = true;
			}
			//save(liY_new);

			if (!b)
				for (int i = 0; i < liX.size(); i++) { // условие выхода из цикла
	
					List<Double> liDiff = new ArrayList<>(n); // вектор разности _Y and _Y*
					for (int j = 0; j < n; j++) liDiff.add(liY_new.get(j) - liX.get(i).get(j));
	
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
		save(liY_new);
	}

	/**
	 * Считает норму матрицы
	 * @param w_temp матрицы
	 * @return её норма
	 */
	private static double normMatrix(List<ArrayList<Double>> w_temp) {
		double summ = 0;
		for (int i = 0; i < w_temp.size(); i++)
			for (int j = 0; j < w_temp.get(i).size(); j++) {
				summ += Math.pow(w_temp.get(i).get(j), 2);
			}
		return Math.sqrt(summ);
	}

	/**
	 * считает средние от вектора
	 * @param liY входной вектор
	 * @return его среднее
	 */
	private static double avg(List<Integer> liY) {
		double summ = 0;
		for (Integer i : liY)
			summ += i;
		return summ / liY.size();
	}

	/**
	 * Евклидова норма Sqrt[ Summ[i*i, {i, 0, n*n}] ]
	 * @param liRES входной вектор
	 * @return его норма
	 */
	private static double norm(List<Double> liRES) {
		double summ = 0;
		for (Double i : liRES) summ += Math.pow(i, 2);
		return Math.sqrt(summ);
	}

	/**
	 * Обучение методом Дельта-проекций
	 */
	private static void learnDeltaProjection() {

		List<ArrayList<Double>> W_old = new ArrayList<ArrayList<Double>>(); // инициализируем матрицу W_old для хранения предыдущего шага
		for (int i = 0; i < n; i++) {
			List<Double> liRow = new ArrayList<>(n);
			for (int j = 0; j < n; j++)
				liRow.add(0.0);
			W_old.add((ArrayList<Double>) liRow);
		}
		List<ArrayList<Double>> W_new = new ArrayList<ArrayList<Double>>(); // инициализируем матрицу W_new для хранения текущего шага
		for (int i = 0; i < n; i++) {
			List<Double> liRow = new ArrayList<>(n);
			for (int j = 0; j < n; j++)
				liRow.add(0.0);
			W_new.add((ArrayList<Double>) liRow);
		}
		List<ArrayList<Double>> W_temp = new ArrayList<ArrayList<Double>>(); // инициализируем матрицу W_temp для хранения W_new - W_old для посчёдта нормы W_new - W_old
		for (int i = 0; i < n; i++) {
			List<Double> liRow = new ArrayList<>(n);
			for (int j = 0; j < n; j++)
				liRow.add(0.0);
			W_temp.add((ArrayList<Double>) liRow);
		}

		List<Double> liTemp = new ArrayList<>(n); // Просто вспомогательный вектор для расчётов
		for (int j = 0; j < n; j++)
			liTemp.add(0.0);

		while (true) {
			for (int t = 0; t < imageName.length; t++) { // предявим каждый обучающий образ
				for (int i = 0; i < n; i++)
					for (int j = 0; j < n; j++)
						W_old.get(i).set(j, W_new.get(i).get(j)); // копирование матрицы с предыдущего шага

				/* Подсчёт выражения W_i-1 * x_i */
				for (int i = 0; i < n; i++) {
					double summ = 0;
					for (int j = 0; j < n; j++)
						summ += W_old.get(i).get(j) * liX.get(t).get(j);
					liTemp.set(i, summ);
				}

				/* Подсчёт x_i - W_i-1 * x_i */
				for (int i = 0; i < n; i++)
					liTemp.set(i, liX.get(t).get(i) - liTemp.get(i));

				/* Подсчёт W_i-1 + (x_i - W_i-1 * x_i) * x_i^T * ny / n и запишем это в W_new */
				for (int i = 0; i < n; i++)
					for (int j = 0; j < n; j++)
						W_new.get(i).set(j, W_old.get(i).get(j) + ny * (liTemp.get(i) * liX.get(t).get(j)) / n);
			}

			for (int i = 0; i < n; i++) // составим матрицу W_new - W_old для подсчёта нормы
				for (int j = 0; j < n; j++)
					W_temp.get(i).set(j, W_new.get(i).get(j) - W_old.get(i).get(j));
			if (normMatrix(W_temp) <= EPS) break; // условие выхода - до стабилизации значения
		}

		// перегоним всё это в W
		for (int i = 0; i < W_old.size(); i++)
			for (int j = 0; j < W_old.get(i).size(); j++)
				W.get(i).set(j, W_new.get(i).get(j));
		System.out.println("Обучение закончнео, матрица весов стабилизировалась");
	}

	/**
	 * Обучение методом проекций
	 */
	private static void learnProjection() {

		List<ArrayList<Double>> W_old = new ArrayList<ArrayList<Double>>(); // инициализируем матрицу W_old для хранения предыдущего шага
		for (int i = 0; i < n; i++) {
			List<Double> liRow = new ArrayList<>(n);
			for (int j = 0; j < n; j++)
				liRow.add(0.0);
			W_old.add((ArrayList<Double>) liRow);
		}
		List<ArrayList<Double>> W_new = new ArrayList<ArrayList<Double>>(); // инициализируем матрицу W_new для хранения текущего шага
		for (int i = 0; i < n; i++) {
			List<Double> liRow = new ArrayList<>(n);
			for (int j = 0; j < n; j++)
				liRow.add(0.0);
			W_new.add((ArrayList<Double>) liRow);
		}

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

			for (int i = 0; i < W_old.size(); i++)
				for (int j = 0; j < W_old.get(i).size(); j++)
					W_old.get(i).set(j, W_new.get(i).get(j)); // копирование матрицы с предыдущего шага

			/* посчитаем знаменатель x_i^T * x_i - x_i^T * W_i-1 * x_i */
			double denominator = internalScalarProduct(liX.get(t)); // выдаст: x_i^T * x_i
			for (int i = 0; i < n; i++) {
				double summ = 0;
				for (int j = 0; j < n; j++) { // подсчёт i-ой компоненты вектора x_i^T * W_i-1
					summ += liX.get(t).get(j) * W_old.get(j).get(i);
				}
				denominator -= liX.get(t).get(i) * summ; // покомпонентный вычет из x_i^T * x_i компоненты вектора x_i^T * W_i-1 умноженой на соответствующую компоненту вектора x_i
			}

			/* теперь посчитаем вектор W_i-1 * x_i - x_i */
			List<Double> liTemp = new ArrayList<>(n);
			for (int i = 0; i < n; i++) {
				double temp = 0;
				for (int j = 0; j < n; j++)
					temp += W_old.get(i).get(j) * liX.get(t).get(j);
				liTemp.add(temp);
			} // только что посчитано выражение W_i-1 * x_i
			// теперь вычтем из W_i-1 * x_i вектор x_i
			for (int i = 0; i < n; i++)
				liTemp.set(i, liTemp.get(i) - liX.get(t).get(i));
			// получили вектор W_i-1 * x_i - x_i

			/* посчитаем матрицу W_i покомпонентно, т.е. W_i[ij] = W_i-1[ij] + (W_i-1 * x_i - x_i)[i] * (W_i-1 * x_i - x_i)[j] / (x_i^T * x_i - x_i^T * W_i-1 * x_i) */
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					W_new.get(i).set(j, W_old.get(i).get(j) + liTemp.get(i) * liTemp.get(j) / denominator);
				}
			}
		}

		// перегоним всё это в W
		for (int i = 0; i < W_old.size(); i++)
			for (int j = 0; j < W_old.get(i).size(); j++)
				W.get(i).set(j, W_new.get(i).get(j));
	}

	/**
	 * Внутреннее скалярное произведение: y_i^T * y_i
	 * @param liY вектор-столбец
	 * @return результат скалярного произведения
	 */
	private static double internalScalarProduct(List<Double> liY) {
		double summ = 0;
		for (int i = 0; i < liY.size(); i++) {
			summ += Math.pow(liY.get(i), 2);
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
					W.get(i).set(j, W.get(i).get(j) + (liX.get(t).get(i) * liX.get(t).get(j)) / n);
				W.get(i).set(i, 0.0);
			}
	}

	/**
	 * Обобщённое правило Хебба
	 */
	private static void learnStandart() { // W = Summ(x_k^t * x_k, {k, 1, imageName.length})
		for (int i = 0; i < imageName.length; i++) {
			add(W, liX.get(i));
		}
	}

	/**
	 * Перемножает вектор x_k сам на себя, создавая матрицу (x_k^T * x_k)
	 * Чистит диагональ
	 * Складывает с матрицей W (w2)
	 * @param w2 матрица W
	 * @param aL вектор x_k из обучающей выборки
	 */
	private static void add(List<ArrayList<Double>> w2, ArrayList<Double> aL) {
		// перемножим : x_k^T * x_k и сразу сложим с w2 и почистим диагональ на каждом шаге, чтоб быстрее было
		// в результате получим симметричную матрицу A^T = A
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++)
				w2.get(i).set(j, w2.get(i).get(j) + aL.get(i) * aL.get(j));
			w2.get(i).set(i, 0.0);
		}
	}

	/**
	 * обнулим и зададим размер матрице W
	 */
	private static void initW() {
		for (int i = 0; i < n; i++) {
			List<Double> liRow = new ArrayList<>(n);
			for (int j = 0; j < n; j++)
				liRow.add(0.0);
			W.add((ArrayList<Double>) liRow);
		}
	}

	/**
	 * Вывод матрицы W в удобном для перегона в Вольфрам виде (для проверки)
	 */
	private static void writeMatrix(List<ArrayList<Integer>> matrix) {
		System.out.print("{");
		for (int i = 0; i < n; i++) {
			System.out.print("{" + W.get(i).get(0));
			for (int j = 1; j < n; j++)
				System.out.print("," + W.get(i).get(j));
			if (i != n - 1)
				System.out.println("},");
			else
				System.out.print("}");
		}
		System.out.println("}");
	}

	/**
	 * Вывод вектора на экран в виде [-1, 1] в удобном для перегона в Вольфрам виде (для проверки)
	 * @param y2 
	 */
	private static void writeVector(List<Integer> y2) {
		System.out.print("{" + y2.get(0));
		for (int i = 1; i < y2.size(); i++) {
			System.out.print(","+ y2.get(i));
		}
		System.out.println("}");
	}

	/**
	 * Переведём картинки обучающих выборок в массивы [-1, 1], запишем их в liX
	 */
	private static void loadImg() {
		for (int i = 0; i < imageName.length; i++) { // пройдёмся по всем именам картинок
			List<Double> liTemp = load(imageName[i]);
			liX.add((ArrayList<Double>) liTemp);
		}
	}

	/**
	 * Загрузка каритнки, усредняем цвет для перевода его в оттенки серого
	 * @param sValue имя картинки, лежащей в ".\\src\\main\\resources\\"
	 * @return
	 */
	private static List<Double> load(String sValue) {
		BufferedImage img = null;
		try {
		    img = ImageIO.read(new File(".\\src\\main\\resources\\" + sValue));
		    n = img.getHeight() * img.getWidth();
		} catch (IOException e) {
			System.err.println("Не удалось найти файл " + sValue);
		}
		// перегоним картинки в массивы {-1, 1} // -1 Black : 1 White
		List<Double> liTemp = new ArrayList<>();
		for (int x = 0; x < img.getHeight(); x++)
			for (int y = 0; y < img.getWidth(); y++) {
				int clr = img.getRGB(y, x); // достаём цвет пикселя
				int  red = (clr & 0x00ff0000) >> 16; // смотрим содержание
				int  green = (clr & 0x0000ff00) >> 8;
				int  blue  =  clr & 0x000000ff;
				int avg = (red + green + blue) / 3;
				liTemp.add(avg * 2.0 / 255 - 1);
			}
		return liTemp;
	}

	/**
	 * Вывод вектора в изображение
	 * @param liY_new входящий вектор
	 */
	private static void save(List<Double> liY_new) {
		BufferedImage img = new BufferedImage((int) Math.sqrt(n), (int) Math.sqrt(n), BufferedImage.TYPE_INT_RGB);
		int k = 0;
		int rgb;
		for (int y = 0; y < Math.sqrt(n); y++)
			for (int x = 0; x < Math.sqrt(n); x++) {
				int temp = (int) ((liY_new.get(k) + 1) * 255 / 2); // перегон из [-1; 1] в [0; 255]
				rgb = (new Color(temp, temp, temp)).getRGB(); // 255 оттенков серого
				img.setRGB(x, y, rgb);
				k++;
			}
		try {
		    File outputfile = new File("outputImg_Step_" + (tye++) + ".png");
		    ImageIO.write(img, "png", outputfile);
		} catch (IOException e) { // когда чёт не пошло
			System.err.println("Так получилось в общем, что кажись файл не захотел создаваться");
		}
	}
}
