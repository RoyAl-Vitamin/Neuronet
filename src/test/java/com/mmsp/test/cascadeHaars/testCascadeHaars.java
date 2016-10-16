package com.mmsp.test.cascadeHaars;

import org.junit.Test;
import static org.junit.Assert.assertEquals;

import org.junit.runner.Description;
import org.junit.runner.JUnitCore;
import org.junit.runner.notification.Failure;
import org.junit.runner.notification.RunListener;

import com.mmsp.neuronet.CascadeHaars;

public class testCascadeHaars {

	@Test
	/**
	 * При заданном масиве интенсивностей изображения считает сумму интенсивностей пикселей произвольного прямоугольника и сравнивает её с эталоном
	 */
	public void equalsIntesity() {
		// http://robocraft.ru/blog/computervision/536.html
		CascadeHaars.intensity = new int[][] {
			{32, 39, 2, 20, 8, 13},
			{15, 14, 11, 12, 13, 14},
			{7, 14, 1, 0, 14, 6},
			{5, 13, 10, 13, 8, 1},
			{0, 2, 12, 20, 15, 7},
			{23, 54, 11, 10, 76, 30},
			{11, 25, 77, 22, 15, 42},
			{45, 33, 65, 3, 17, 8},
			{25, 1, 54, 54, 88, 6}
		};
		CascadeHaars.w = 9;
		CascadeHaars.h = 6;
		CascadeHaars.calcIntegralPic();
		int temp = CascadeHaars.L(2, 4, 5, 8);
		assertEquals(371, temp);
	}

	@Test
	/**
	 * При заданном массиве интенсивностей изображения считает интегральное изображение и сравнивает его с эталоном 
	 */
	public void equalsIntegralPic() {
		CascadeHaars.intensity = new int[][] {
			{32, 39, 2, 20, 8, 13},
			{15, 14, 11, 12, 13, 14},
			{7, 14, 1, 0, 14, 6},
			{5, 13, 10, 13, 8, 1},
			{0, 2, 12, 20, 15, 7},
			{23, 54, 11, 10, 76, 30},
			{11, 25, 77, 22, 15, 42},
			{45, 33, 65, 3, 17, 8},
			{25, 1, 54, 54, 88, 6}
		};
		CascadeHaars.w = 9;
		CascadeHaars.h = 6;
		CascadeHaars.calcIntegralPic();
		int temp[][] = new int[][] {
			{32,  71,  73,  93,  101,  114}, 
			{47,  100, 113, 145, 166,  193}, 
			{54,  121, 135, 167, 202,  235}, 
			{59,  139, 163, 208, 251,  285}, 
			{59,  141, 177, 242, 300,  341}, 
			{82,  218, 265, 340, 474,  545}, 
			{93,  254, 378, 475, 624,  737}, 
			{138, 332, 521, 621, 787,  908}, 
			{163, 358, 601, 755, 1009, 1136}
		};
		assertEquals(temp, CascadeHaars.integralPic);
	}

	public static void main(String[] args) {
		JUnitCore core = new JUnitCore();
		core.addListener(new CalcListener());
		core.run(testCascadeHaars.class);
	}
}

class CalcListener extends RunListener {

	@Override
	public void testStarted(Description desc) {
		System.out.println("Started:" + desc.getDisplayName());
	}
 
	@Override
	public void testFinished(Description desc) {
		System.out.println("Finished:" + desc.getDisplayName());
	}

	@Override
	public void testFailure(Failure fail) {
		System.out.println("Failed:" + fail.getDescription().getDisplayName() + " [" + fail.getMessage() + "]");
	}
}