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
public class MLP_2Class_Classifier_moon extends MLP_2Class_Classifier {


    public static void main(String[] args) throws Exception {
    	
    	doIt (
    			
	        123, 	// seed
	        0.01, 	// learningRate
	        50, 	// batchSize
	        100, 	// nEpochs
	
	        2, 	// numInputs
	        2, 	// numOutputs
	        20, // numHiddenNodes

	        // moon
	        new File("src/main/resources/classification/moon_data_train.csv"), // trainingData
	        new File("src/main/resources/classification/moon_data_eval.csv"), // testData        
	        -1.5, // xMin
	        2.5, // xMax
	        -1, // yMin
	        1.5 // yMax
        );
    }
}