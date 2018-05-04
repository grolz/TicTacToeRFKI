package de.hbo.dl4j.classifier.samples;

import org.nd4j.linalg.io.ClassPathResource;

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
public class MLP_2Class_Classifier_saturn extends MLP_2Class_Classifier {


    public static void main(String[] args) throws Exception {
    	
    	doIt (
    			
	        123, 	// seed
	        0.01, 	// learningRate
	        50, 	// batchSize
	        100, 	// nEpochs
	
	        2, 	// numInputs
	        2, 	// numOutputs
	        20, // numHiddenNodes
	        
	        // saturn
	        new ClassPathResource("/classification/saturn_data_train.csv").getFile(), // trainingData
	        new ClassPathResource("/classification/saturn_data_eval.csv").getFile(), // testData
	        -15, // xMin
	        15, // xMax
	        -15, // yMin
	        15 // yMax
        );
    }
}