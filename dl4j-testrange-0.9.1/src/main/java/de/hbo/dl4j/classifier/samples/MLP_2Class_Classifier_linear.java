package de.hbo.dl4j.classifier.samples;

import java.io.File;

import de.hbo.dl4j.classifier.MLP_2Class_Classifier;

/**
 * "Linear" Data Classification Example
 *
 * Based on the data from Jason Baldridge:
 * https://github.com/jasonbaldridge/try-tf/tree/master/simdata
 *
 * @author Josh Patterson
 * @author Alex Black (added plots)
 *
 */
public class MLP_2Class_Classifier_linear extends MLP_2Class_Classifier {


    public static void main(String[] args) throws Exception {
    	
    	doIt (
    			
	        123, 	// seed
	        0.01, 	// learningRate
	        50, 	// batchSize
	        5, 	// nEpochs
	
	        2, 	// numInputs
	        2, 	// numOutputs
	        20, // numHiddenNodes
	
	        // linear
	        new File("src/main/resources/classification/linear_data_train.csv"), // trainingData
	        new File("src/main/resources/classification/linear_data_eval.csv"), // testData
	        0, // xMin
	        1.0, // xMax
	        -0.2, // yMin
	        0.8 // yMax
        );
    }
}