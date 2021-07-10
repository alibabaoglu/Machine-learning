/************************************************************************
 * \brief: Main method reading in the spiral data and learning the       *
 *         mapping using the Least Mean Squares method within a neural   *
 *         network.
 *																		*
 * (c) copyright by Jörn Fischer											*
 *                                                                       *																		*
 * @autor: Prof.Dr.Jörn Fischer											*
 * @email: j.fischer@hs-mannheim.de										*
 *                                                                       *
 * @file : MainFrame.java                                                *
 *************************************************************************/

import java.awt.Color;
import java.awt.Image;
import java.awt.image.ImageObserver;

import javax.swing.JFrame;

/**
 * GNN Aufgabe 4
 * @author Ali Babaoglu - 1827133
 */

@SuppressWarnings("serial")
public class MainFrame extends JFrame {

	public static final int imageWidth = 900;
	public static final int imageHeight = 600;
	public InputOutput inputOutput = new InputOutput(this);
	public boolean stop = false;
	ImagePanel canvas = new ImagePanel();
	ImageObserver imo = null;
	Image renderTarget = null;
	// graphical output_size
	public static final int NEURONS = 9;
	public static final int PIXEL = 9;

	double convWeight[][];
	double out[][];
	int xRes, yRes;
	byte buffer[];

	public MainFrame(String[] args) {
		super("Convolutional Neural Networks");

		convWeight = new double[NEURONS][PIXEL];
		out = new double[3][NEURONS];

		getContentPane().setSize(imageWidth, imageHeight);
		setSize(imageWidth + 100, imageHeight + 100);
		setDefaultCloseOperation(EXIT_ON_CLOSE);
		setVisible(true);

		canvas.img = createImage(imageWidth, imageHeight);

		add(canvas);

		run();
	}

	/**
	 * @brief: run method calls my Main and puts the results on the screen
	 */
	public void run() {

		myMain();

		repaint();
		setVisible(true);
		do {
		} while (!stop);

		dispose();
	}

	/**
	 * Construct main frame
	 *
	 * @param args passed to MainFrame
	 */
	public static void main(String[] args) {
		new MainFrame(args);
	}


	// --- computeConvolution() ----------------------------------
	void computeConvolution(int x, int y) {
		double sum;
		out[0][0] = getPixel(x - 1, y - 1);
		out[0][1] = getPixel(x, y - 1);
		out[0][2] = getPixel(x + 1, y - 1);

		out[0][3] = getPixel(x - 1, y);
		out[0][4] = getPixel(x, y);
		out[0][5] = getPixel(x + 1, y);

		out[0][6] = getPixel(x - 1, y + 1);
		out[0][7] = getPixel(x, y + 1);
		out[0][8] = getPixel(x + 1, y + 1);

		// --- activate first convolutional layer with ReLu output
		//  Here, neurons with the same weights are pushed over the image as a mask. The pixel values are multiplied by the weights and then summed up.
		for (int i = 0; i < PIXEL; i++) {
			sum= 0;
			for (int j = 0; j <NEURONS ; j++) {
				sum+= out[0][j] * convWeight[j][i];

			}
			// Sum should only be saved if greater than 0. Otherwise save 0
			out[1][i] = Math.max(0.0,sum);

		}

	}

	// --- adaptWeights() ----------------------------------------
	void adaptWeights(int x, int y) {
		double sum;
		// --- backwards activation
		for (int pixel = 0; pixel < PIXEL; pixel++) {
			sum = 0;
			for (int neuron = 0; neuron < NEURONS; neuron++) {
				sum += out[1][neuron] * convWeight[neuron][pixel];
			}
			out[2][pixel] = sum > 0 ? sum : 0;
		}
		// --- training layer with Constrastive Divergence
		for (int neuron = 0; neuron < NEURONS; neuron++) {
			for (int pixel = 0; pixel < PIXEL; pixel++) {
				convWeight[neuron][pixel] -= 0.01 * (out[2][pixel] - out[0][pixel]) * out[1][neuron];
			}
		}
	}

	// --- displayFilters() ----------------------------
	void displayFilters(int xOffset) {
		int offset = xRes + xOffset;
		double col = 0;
		for (int i = 0; i < NEURONS; i++) {
			col = (convWeight[i][0] + 0.5) * 255 % 255 < 0 ? 0 : (convWeight[i][0] + 0.5) * 255 % 255;
			inputOutput.fillRect(offset, 10 + i * 40, 9, 9, new Color((int) col, (int) col, (int) col));
			col = (convWeight[i][1] + 0.5) * 255 % 255 < 0 ? 0 : (convWeight[i][1] + 0.5) * 255 % 255;
			inputOutput.fillRect(offset + 10, 10 + i * 40, 9, 9, new Color((int) col, (int) col, (int) col));
			col = (convWeight[i][2] + 0.5) * 255 % 255 < 0 ? 0 : (convWeight[i][2] + 0.5) * 255 % 255;
			inputOutput.fillRect(offset + 20, 10 + i * 40, 9, 9, new Color((int) col, (int) col, (int) col));

			col = (convWeight[i][3] + 0.5) * 255 % 255 < 0 ? 0 : (convWeight[i][3] + 0.5) * 255 % 255;
			inputOutput.fillRect(offset, 20 + i * 40, 9, 9, new Color((int) col, (int) col, (int) col));
			col = (convWeight[i][4] + 0.5) * 255 % 255 < 0 ? 0 : (convWeight[i][4] + 0.5) * 255 % 255;
			inputOutput.fillRect(offset + 10, 20 + i * 40, 9, 9, new Color((int) col, (int) col, (int) col));
			col = (convWeight[i][5] + 0.5) * 255 % 255 < 0 ? 0 : (convWeight[i][5] + 0.5) * 255 % 255;
			inputOutput.fillRect(offset + 20, 20 + i * 40, 9, 9, new Color((int) col, (int) col, (int) col));

			col = (convWeight[i][6] + 0.5) * 255 % 255 < 0 ? 0 : (convWeight[i][6] + 0.5) * 255 % 255;
			inputOutput.fillRect(offset, 30 + i * 40, 9, 9, new Color((int) col, (int) col, (int) col));
			col = (convWeight[i][7] + 0.5) * 255 % 255 < 0 ? 0 : (convWeight[i][7] + 0.5) * 255 % 255;
			inputOutput.fillRect(offset + 10, 30 + i * 40, 9, 9, new Color((int) col, (int) col, (int) col));
			col = (convWeight[i][8] + 0.5) * 255 % 255 < 0 ? 0 : (convWeight[i][8] + 0.5) * 255 % 255;
			inputOutput.fillRect(offset + 20, 30 + i * 40, 9, 9, new Color((int) col, (int) col, (int) col));
		}
	}

	public void myMain() {
		FileIO inFile = new FileIO("exe.ppm");
		readImage(inFile);

		for (int y = 0; y < yRes; y++) {
			for (int x = 0; x < xRes; x++) {
				int col = (int) (255.0 * getPixel(x, y));
				if (col < 0) col = 0;//255+col;
				if (col > 255) col = 255;
//				System.out.println(col + ",");

				inputOutput.drawPixel(x, y, new Color(col, col, col));
			}
		}

		// initialize weights
		for (int t = 0; t < NEURONS; t++) {
			for (int i = 0; i < PIXEL; i++) {
				convWeight[t][i] = Math.random() / 10.0;
			}
		}


		for (int t = 0; t < 10; t++) {
			int stride = 1;

			displayFilters(20 + t * 50);

			for (int y = 3; y < yRes - 3; y += stride) {
				for (int x = 3; x < xRes - 3; x += stride) {
					computeConvolution(x, y);
					adaptWeights(x, y);
				}
			}

		}
		repaint();
		setVisible(true);

	}

	// --- readLine --------------------------------------
	void readLine(FileIO inFile) {
		byte c;
		do {
			c = inFile.fgetc();
		} while (c != '\n');
	}

	// --- readImage -------------------------------------
	void readImage(FileIO inFile) {

		char c;
		String text = "";
		readLine(inFile);
		readLine(inFile);
		for (int t = 0; (c = (char) inFile.fgetc()) != ' '; t++) {
			text += c;
		}

		xRes = (int) Float.parseFloat(text);
		text = "";
		for (int t = 0; (c = (char) inFile.fgetc()) != '\n'; t++) {
			text += c;
		}
		yRes = (int) Float.parseFloat(text);

		System.out.println("xRes = " + xRes + "  yres = " + yRes);

		readLine(inFile);
		buffer = new byte[xRes * yRes];
		for (int y = 0; y < yRes; y++) {
			for (int x = 0; x < xRes; x++) {
				buffer[x + y * xRes] = inFile.fgetc();
				byte dummy = inFile.fgetc();
				dummy = inFile.fgetc();

			}
		}
	}

	// --- getPixel ------------------------------------
	double getPixel(int x, int y) {
		int xx = (int) (buffer[x + y * xRes]);
		if (xx < 0)
			xx = 255 + xx;
		return xx / 255.0;
	}


}
