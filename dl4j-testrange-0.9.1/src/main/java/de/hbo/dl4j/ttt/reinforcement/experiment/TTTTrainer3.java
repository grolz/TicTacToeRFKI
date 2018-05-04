package de.hbo.dl4j.ttt.reinforcement.experiment;

import java.util.Random;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.rl4j.network.dqn.DQN;
import org.deeplearning4j.rl4j.util.Constants;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.lossfunctions.LossFunctions;

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
public class TTTTrainer3 {
	
	private static int testNumber;
	private static float[][] testFeatures;
	private static float[]   testLabels;

    public static void main(String[] args) throws Exception {
    	
    	Random random = new Random();
    	
        float[] numbers = new float[] {-1, 0, 1};
        float[] fields = new float[]  {.1f, .2f, .3f, .4f};

        
        testNumber = 1000;
        testFeatures = new float[testNumber][fields.length];
        testLabels   = new float[testNumber];
        
        for (int j = 0; j < testNumber; j++) {
            
	        float[] features = new float[fields.length];
	        float label = 0;
	        for (int i = 0; i < features.length; i++) {
	        	
	        	int idx = random.nextInt(numbers.length);
	        	
	        	features[i] = numbers[idx];
	        	label += numbers[idx]*fields[i];
	        }
        	
        	testFeatures[j] = features;
        	testLabels[j] = label;
    	}
 
        // ------------------------------------------------------------------------------------------------------------------
        // MODEL SETUP
        System.out.print("Setup model: ");
        
        MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
    		.seed(Constants.NEURAL_NET_SEED)
            .iterations(1)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(0.01)
            .updater(Updater.NESTEROVS).momentum(0.9)
            //.updater(Updater.ADAM)
            .weightInit(WeightInit.XAVIER)
            .regularization(true)
            .l2(0.01)
            .list()
            
            
            //original:
//			.layer(0, new DenseLayer.Builder().nIn(fields.length).nOut(16).activation(Activation.RELU).build())
//        	.layer(1, new DenseLayer.Builder().nIn(16).nOut(16).activation(Activation.RELU).build())
//        	.layer(2, new DenseLayer.Builder().nIn(16).nOut(16).activation(Activation.RELU).build())
//			.layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(16).nOut(1).build())
			
            //diese sehr gut:
//			.layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(fields.length).nOut(1).build())

            
			.layer(0, new DenseLayer.Builder().nIn(fields.length).nOut(16).activation(Activation.RELU).build())
			.layer(1, new DenseLayer.Builder().nIn(16).nOut(16).activation(Activation.RELU).build())
			.layer(2, new DenseLayer.Builder().nIn(16).nOut(16).activation(Activation.RELU).build())
			.layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(16).nOut(1).build())
            
            
            .pretrain(false)
			.backprop(true)
			.build()
		);
        
		model.init();
		
        DQN<?> dqn = new DQN<>(model);

        System.out.println("done");

        // ------------------------------------------------------------------------------------------------------------------
        // TRAINING
        
        System.out.println("Train model ...");
        System.out.println();

        int epoch = 0;
        int evalEpoch = 500;
        int maxepoch = 1000000;
        
        double startError = evaluation(dqn);
        System.out.println("[" + epoch + "]: \t" + startError);

        while (++epoch <= maxepoch) {
        
        	// training
        	{
		        float[] features = new float[fields.length];
		        float label = 0;
		        
		        for (int i = 0; i < features.length; i++) {
		        	
		        	int idx = random.nextInt(numbers.length);
		        	
		        	features[i] = numbers[idx];
		        	label += numbers[idx]*fields[i];
		        }
	        	
	        	dqn.fit(new NDArray(features), new NDArray(new float[] {label}));
        	}
	        
        	// evaluation
	        if (epoch >= evalEpoch && epoch%evalEpoch==0) {
	        	double currentError = evaluation(dqn);
	        	System.out.print("[" + epoch + "]: \t" + currentError + " \t(" + (currentError-startError) + ")\r");
	        }
        }
   
        System.out.println("... done.");
    }
    
    private static double evaluation(DQN<?> dqn) {
    	double mse = 0d;
		for (int j = 0; j < testNumber; j++) mse += Math.pow(dqn.output(new NDArray(testFeatures[j])).data().asDouble()[0] - testLabels[j], 2d);
    	return mse/testNumber;
    }
}