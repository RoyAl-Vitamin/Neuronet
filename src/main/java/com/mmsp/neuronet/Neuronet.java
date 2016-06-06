package com.mmsp.neuronet;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

public class Neuronet {

	static String[] imageName = {"А.png", "Б.png", "И.png", "Н.png"}; // имена учебных выборок

	static List<ArrayList<Integer>> liX = new ArrayList<ArrayList<Integer>>(); // массив массива разложенных по цвету картинок (обучающих выборок)

	static List<ArrayList<Integer>> W = new ArrayList<ArrayList<Integer>>(); // Матрица W = Summ(x_k^t * x_k, {k, 1, imageName.length})

	static List<Integer> Y = null; // вектор с шумами, который будем распознавать

	static int n; // порядок матрицы W или длина любого вектора из массива liX, или длина ветора Y

	static int clBlack = (new Color(0, 0, 0)).getRGB();

	static int clWhite = (new Color(255, 255, 255)).getRGB();

	static double EPS = 1E-2;

	public static void main(String[] args) {
		loadImg();
		createW();

		Y = load("Y.png");
		List<Integer> liTemp = new ArrayList<>(n);
		for (Integer i : Y) liTemp.add(i); // скопируем вектор Y

		/*for (int i = 0; i < imageName.length; i++) // Выведем ветора, посмотрим чё там
			writeVector(liX.get(i));
		writeVector(Y);*/

		double minNorm = EPS + 1;
		int k = 0;
		while (true) {
			// перемножаем W и Y
			for (int i = 0; i < n; i++) { // бежим по строкам матрицы W
				int summ = 0;
				for (int j = 0; j < n; j++) { // по компонентам вектора Y
					summ += W.get(i).get(j) * liTemp.get(j);
				}
				// Y = sign(Y)
				if (summ >= 0) liTemp.set(i, 1); else liTemp.set(i, -1);
			}

			/*List<Integer> liRES = new ArrayList<>(n);
			for (int i = 0; i < n; i++) liRES.add(liTemp.get(i) - Y.get(i));
			System.err.println(norm(liRES));*/ // Проверка неравенста векторов

			boolean b = false;

			for (int i = 0; i < liX.size(); i++) { // условие выхода из цикла

				List<Integer> liDiff = new ArrayList<>(n); // вектор разности _Y and _Y*
				for (int j = 0; j < n; j++) liDiff.add(liTemp.get(j) - liX.get(i).get(j));

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
		save(liTemp);
	}

	private static double norm(List<Integer> arrayList) { // Евклидова норма Sqrt[ Summ[i*i, {i, 0, n*n}] ]
		double summ = 0;
		for (Integer i : arrayList) summ += Math.pow(i, 2);
		return Math.sqrt(summ);
	}

	/**
	 * Соберём матрицу W
	 */
	private static void createW() {
		initW();
		for (int i = 0; i < imageName.length; i++) {
			add(W, liX.get(i));
		}
		//writeMatrix(W);
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
	 * Перемножает вектор x_k сам на себя, создавая матрицу (x_k^T * x_k)
	 * Чистит диагональ
	 * Складывает с матрицей W (w2)
	 * @param w2 матрица W
	 * @param aL вектор x_k из обучающей выборки
	 */
	private static void add(List<ArrayList<Integer>> w2, ArrayList<Integer> aL) {
		// перемножим : x_k^T * x_k и сразу сложим с w2 и почистим диагональ на каждом шаге, чтоб быстрее было
		// в результате получим симметричную матрицу A^T = A
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++)
				w2.get(i).set(j, w2.get(i).get(j) + aL.get(i) * aL.get(j));
			w2.get(i).set(i, 0);
		}
	}

	/**
	 * обнулим и зададим размер матрице W
	 */
	private static void initW() {
		for (int i = 0; i < n; i++) {
			List<Integer> liRow = new ArrayList<>(n);
			for (int j = 0; j < n; j++)
				liRow.add(0);
			W.add((ArrayList<Integer>) liRow);
		}
	}

	/**
	 * Вывод вектора на экран в виде {-1, 1} в удобном для перегона в Вольфрам виде (для проверки)
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
	 * Переведём картинки обучающих выборок в массивы {-1, 1}, запишем их в liX
	 */
	private static void loadImg() {
		for (int i = 0; i < imageName.length; i++) { // пройдёмся по всем именам картинок
			List<Integer> liTemp = load(imageName[i]);
			liX.add((ArrayList<Integer>) liTemp);
		}
	}

	private static List<Integer> load(String sValue) {
		BufferedImage img = null;
		try {
		    img = ImageIO.read(new File(".\\src\\main\\resources\\" + sValue));
		    n = img.getHeight() * img.getWidth();
		} catch (IOException e) {
			System.err.println("Не удалось найти файл " + sValue);
		}
		// перегоним картинки в массивы {-1, 1} // -1 Black : 1 White
		List<Integer> liTemp = new ArrayList<>();
		for (int x = 0; x < img.getHeight(); x++)
			for (int y = 0; y < img.getWidth(); y++) {
				int clr = img.getRGB(y, x); // достаём цвет пикселя
				int  red = (clr & 0x00ff0000) >> 16; // смотрим содержание
				int  green = (clr & 0x0000ff00) >> 8;
				int  blue  =  clr & 0x000000ff;
				if (red == 0 && green == 0 && blue == 0) {	// Чёрный
					liTemp.add(-1);
				} else {									// Белый
					liTemp.add(1);
				}
			}
		return liTemp;
	}

	/**
	 * Вывод вектора в изображение
	 * @param outF входящий вектор
	 */
	private static void save(List<Integer> outF) {
		BufferedImage img = new BufferedImage((int) Math.sqrt(n), (int) Math.sqrt(n), BufferedImage.TYPE_INT_RGB);
		int k = 0;
		int rgb;
		for (int y = 0; y < Math.sqrt(n); y++)
			for (int x = 0; x < Math.sqrt(n); x++) {
				if (outF.get(k) == -1) rgb = clBlack; else rgb = clWhite;
				img.setRGB(x, y, rgb);
				k++;
			}
		try {
		    File outputfile = new File("outputImg.png");
		    ImageIO.write(img, "png", outputfile);
		} catch (IOException e) { // когда чёт не пошло
			System.err.println("Так получилось в общем, что кажись файл не захотел создаваться");
		}
	}
}
