package com.mmsp.neuronet;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

/**
 * Каскад Хаара
 * Находит нужные изображения на картинке и масштабирует их
 * @author Alex
 *
 */
public class CascadeHaars {

	// public for Test
	public static int intensity[][] = null; // Матрица интенсивностей
	public static int integralPic[][] = null; // Интегральное представление изображения

	public static int w = 0;
	public static int h = 0;

	public static void main(String[] args) {
		intensity = load(""); // TODO Вставить путь до изображения
		calcIntegralPic();

		/*
		 * Далее гонять окно меняющегося размера по готовым выделенным областям и считать свёртки S=X/Y (для учёта масштаба), 
		 * где Х - L т тёмной области признака Хаара, У - L от светлой области
		 * в дальнейшем на исходном изображении для распознавании так же ездить окном разного размера
		 * и считать свёртки, сравнивая их с уже расчитанными
		 */
	}

	public static void calcIntegralPic() {
		integralPic = new int[w][h];
		integralPic[w - 1][h - 1] = L(h - 1, w - 1); // Инициализация матрицы рекурсивным способом
	}

	/**
	 * Считает сумму пикселей произвольного прямоугольника 
	 * @param x1 координата по х для правой нижней вершины прямоугольника А
	 * @param y1 координата по у для правой нижней вершины прямоугольника А
	 * @param x2 координата по х для правой нижней вершины прямоугольника С
	 * @param y2 координата по х для правой нижней вершины прямоугольника С
	 * @return Интегральную сумму
	 */
	public static int L(int x1, int y1,int x2, int y2) {
		// S(ABCD) = L(A) + L(С) — L(B) — L(D)
		// +---+---+
		// | A | B |
		// +---+---+
		// | D | C |
		// +---+---+
		//return L(x1, y1) + L(x2, y2) - L(x2, y1) - L(x1, y2); // Для не инициализированной матрицы integralPic[][]
		return integralPic[y1][x1] + integralPic[y2][x2] - integralPic[y1][x2] - integralPic[y2][x1]; // Для инициализированной
	}

	/**
	 * Заполняет матрицу интегрального представления изображения рекурсивно
	 * @param length
	 * @param length2
	 */

	private static int L(int x, int y) {
		if (x >= 0 && y >= 0) {
			integralPic[y][x] = intensity[y][x] - L(x - 1, y - 1) + L(x, y - 1) + L(x - 1, y);
			return integralPic[y][x];
		}
		return 0;
	}


	/**
	 * Загрузка каритнки и перегон её в 2-ух мерный массив интенсивностей
	 * Для серых изображений от 0 до 255
	 * Для цветных от 0 до 255 * 3 (RGB) (это что бы 0 не похерил всю матрицу)
	 * @param sValue имя картинки, лежащей в ".\\src\\main\\resources\\"
	 * @return 
	 */
	private static int[][] load(String sValue) {
		BufferedImage img = null;
		try {
			img = ImageIO.read(new File(".\\src\\main\\resources\\" + sValue));
			w = img.getWidth();
			h = img.getHeight();
		} catch (IOException e) {
			System.err.println("Не удалось найти файл " + sValue);
		}

		// перегоним картинки в массивы {-1, 1} // -1 Black : 1 White
		int arTemp[][] = new int[w][h];
		for (int x = 0; x < img.getWidth(); x++)
			for (int y = 0; y < img.getHeight(); y++) {
				int clr = img.getRGB(y, x); // достаём цвет пикселя
				int  red = (clr & 0x00ff0000) >> 16; // смотрим содержание
				int  green = (clr & 0x0000ff00) >> 8;
				int  blue  =  clr & 0x000000ff;
				arTemp[x][y] = red + green + blue;
			}
		return arTemp;
	}

	/**
	 * Считает интегральное представление изображения
	 * @param x координата по длине
	 * @param y по ширине
	 * @return Summ (I(i,j), {i=0,x, j=0,y}), I(i,j) - интенсивность данного пикселя (яркость пикселя исходного изображения)
	 * @throws Exception вышли за пределы изображения
	 */
	private int II(int x, int y) throws Exception {
		if (x >= intensity.length) throw new Exception("x > Ширины");
		if (y >= intensity[x].length) throw new Exception("y > Длины");
		int summ = 0;
		for (int i = 0; i < x; i++)
			for (int j = 0; j < y; j++) {
				summ += intensity[i][j];
			}
		return summ;
	}

	public Object getSum(int i, int j) {
		return 51;
	}
}
