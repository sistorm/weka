/**
 * 
 */
package org.ets.gti770.weka_lab;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.InvalidPathException;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;

/**
 * @author charly
 *
 */
public class Main {

	private static final boolean DEBUG = true;

	private static String TRAINNING_FILE_PATH = "./data/spamdata-dev.arff";

	/**
	 * @param args
	 * 
	 */
	public static void main(String[] args) throws Exception {

		// load data

		Instances train = loadInstaces(TRAINNING_FILE_PATH);
		train.setClassIndex(train.numAttributes() - 1);

		Instances test;
		try {
			test = loadInstaces(args[0]);
			test.setClassIndex(test.numAttributes() - 1);
		} catch (Exception ex) {
			throw new InvalidPathException("Invalid input file for validation (parameter 1)", ex.getMessage());
		}
		if (!train.equalHeaders(test))
			throw new IllegalArgumentException("Train and test set are not compatible: " + train.equalHeadersMsg(test));

		// train classifier
		Classifier IBkClassifier = pepareIBkClassifier();
		IBkClassifier.buildClassifier(train);

		Classifier j48Classifier = pepareJ48Classifier();
		j48Classifier.buildClassifier(train);

		Classifier naiveBayesClassifier = pepareNaiveBayesClassifier();
		naiveBayesClassifier.buildClassifier(train);

		Classifier moreEffectiveClassifier = naiveBayesClassifier;
		Classifier lessEffectiveClassifier = IBkClassifier;

		// output predictions

		if (DEBUG) {
			PrintStream output = System.out;
			printPrediction(IBkClassifier, test, output, IBkClassifier.getClass().getName() + " - ");
			waitUser();
			printPrediction(j48Classifier, test, output, j48Classifier.getClass().getName() + " - ");
			waitUser();
			printPrediction(naiveBayesClassifier, test, output, naiveBayesClassifier.getClass().getName() + " - ");

			waitUser();
			printPrediction(moreEffectiveClassifier, test, output,
					"MoreEffective - " + moreEffectiveClassifier.getClass().getName() + " - ");
			waitUser();
			printPrediction(lessEffectiveClassifier, test, output,
					"LessEffective - " + lessEffectiveClassifier.getClass().getName() + " - ");

		} else {
			PrintStream outputMoreEffectiveClassifier = null;
			try {
				File f = new File(args[1]);
				FileOutputStream fos = new FileOutputStream(f);
				outputMoreEffectiveClassifier = new PrintStream(fos);

			} catch (ArrayIndexOutOfBoundsException | NullPointerException | FileNotFoundException ex) {
				throw new InvalidPathException("Invalid output file for More Effective classifier (parameter 2)",
						ex.getMessage());
			}
			printPrediction(moreEffectiveClassifier, test, outputMoreEffectiveClassifier, "");

			PrintStream outputLessEffectiveClassifier = null;
			try {
				File f = new File(args[2]);
				FileOutputStream fos = new FileOutputStream(f);
				outputLessEffectiveClassifier = new PrintStream(fos);
			} catch (ArrayIndexOutOfBoundsException | NullPointerException | FileNotFoundException ex) {
				throw new InvalidPathException("Invalid output file for More Effective classifier  (parameter 3)",
						ex.getMessage());
			}

			printPrediction(lessEffectiveClassifier, test, outputLessEffectiveClassifier, "");
		}

	}

	private static void waitUser() throws IOException {
		System.in.read();
		while (System.in.available() > 0)
			System.in.read();

	}

	private static Instances loadInstaces(String location) throws Exception {
		Instances instances = DataSource.read(location);
		return instances;
	}

	private static void printPrediction(Classifier cls1, Instances test, PrintStream outputStream, String prefixOutput)
			throws Exception {
		PrintStream output = outputStream;
		if (output == null)
			output = System.out;

		int totalTestInstances = test.numInstances();
		int validPrediction = 0;
		if (DEBUG)
			output.println(prefixOutput + "# - actual - predicted - error - distribution");

		for (int i = 0; i < totalTestInstances; i++) {
			double pred = cls1.classifyInstance(test.instance(i));
			output.print(prefixOutput);
			if (DEBUG) {
				output.print((i + 1));
				output.print(" - ");
				output.print(test.instance(i).toString(test.classIndex()));
				output.print(" - ");
			}
			output.print(test.classAttribute().value((int) pred));
			if (DEBUG) {
				if (pred == test.instance(i).classValue())
					validPrediction += 1;
			}
			output.println();
		}

		if (DEBUG) {
			double validPredictionPercent = (double) validPrediction / totalTestInstances;
			output.println(prefixOutput + "percentage valid:   " + validPredictionPercent + " %");
			output.println(prefixOutput + "percentage invalid: " + (1 - validPredictionPercent + " %"));
		}

	}

	private static Classifier pepareIBkClassifier() throws Exception {
		IBk ibkClassifier = new IBk();

		ibkClassifier.setKNN(20);
		SelectedTag distanceW = new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING);
		ibkClassifier.setDistanceWeighting(distanceW);
		NearestNeighbourSearch NNSearch = new LinearNNSearch();
		NNSearch.setDistanceFunction(new ManhattanDistance());
		ibkClassifier.setNearestNeighbourSearchAlgorithm(NNSearch);

		return ibkClassifier;
	}

	private static Classifier pepareNaiveBayesClassifier() {
		NaiveBayes NBClassifier = new NaiveBayes();

		NBClassifier.setUseSupervisedDiscretization(true);

		return NBClassifier;
	}

	private static Classifier pepareJ48Classifier() {
		J48 j48Classifier = new J48();

		j48Classifier.setCollapseTree(false);
		j48Classifier.setConfidenceFactor((float) 0.25);
		j48Classifier.setMinNumObj(4);
		j48Classifier.setSubtreeRaising(false);
		j48Classifier.setUseMDLcorrection(false);

		return j48Classifier;
	}
}
