package de.hbo.dl4j.ttt.reinforcement.experiment;

import java.awt.Color;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import de.hbo.diagramm.diagramm2D.PunktDoubleDouble;
import de.hbo.diagramm.diagramm2D.StreckenzugDoubleDouble;
import de.hbo.dl4j.ttt.reinforcement.experiment.Evaluator.DQNEvaluator;
import de.hbo.dl4j.ttt.reinforcement.experiment.Evaluator.PerfectEvaluator;
import de.hbo.tictactoe.TTTGame.TTTPlayer;
import de.hbo.tictactoe.TicTacToe;

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
public class TTTTrainer {
	
    private static final Random random = new Random(123);

    /**
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
 
        // ------------------------------------------------------------------------------------------------------------------
        // MODEL SETUP
        System.out.print("Setup model:");
        
        
        // ******************************************************************************************************************************************************************

//        final String bezeichner = "setting1";
//        final int trainEpochs = 100;
//        final int saveEpochs = 5000;
//        final int startEpoch = 1;
//        final int numberTestMatches = 500;
//        final double epsilon = .25;
//        final double gamma = .25;
//        final int hidden = 32;
//
//        MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
//    		.seed(Constants.NEURAL_NET_SEED)
//            .iterations(20)
//            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//            .learningRate(.1)
//            //.updater(Updater.NESTEROVS).momentum(0.9)
//            //.updater(Updater.RMSPROP).rho(conf.getRmsDecay())//.rmsDecay(conf.getRmsDecay())
//            .updater(Updater.ADAM)
//            .weightInit(WeightInit.XAVIER)
//            .regularization(true)
//            .l2(.01)
//            .list()
//            .layer(0, new DenseLayer.Builder().nIn(18).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(1, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(2, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(3, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(4, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//			.layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(hidden).nOut(1).build())
//			.pretrain(false)
//			.backprop(true)
//			.build()
//		);
        
        // ******************************************************************************************************************************************************************

//        final String bezeichner = "setting2";
//        final int trainEpochs = 100;
//        final int saveEpochs = 5000;
//        final int startEpoch = 1;
//        final int numberTestMatches = 500;
//        final double epsilon = .25;
//        final double gamma = .25;
//        final int hidden = 32;
//
//        MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
//    		.seed(Constants.NEURAL_NET_SEED)
//            .iterations(20)
//            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//            .learningRate(.1)
//            //.updater(Updater.NESTEROVS).momentum(0.9)
//            //.updater(Updater.RMSPROP).rho(conf.getRmsDecay())//.rmsDecay(conf.getRmsDecay())
//            .updater(Updater.ADAM)
//            .weightInit(WeightInit.XAVIER)
//            .regularization(true)
//            .l2(.01)
//            .list()
//            .layer(0, new DenseLayer.Builder().nIn(18).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(1, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(2, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(3, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(4, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(5, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(6, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(7, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(8, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//			.layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(hidden).nOut(1).build())
//			.pretrain(false)
//			.backprop(true)
//			.build()
//		);
        
        // ******************************************************************************************************************************************************************

//        final String bezeichner = "setting3";
//        final int trainEpochs = 100;
//        final int saveEpochs = 5000;
//        final int numberTestMatches = 500;
//        final int startEpoch = 1;
//        final double epsilon = .25;
//        final double gamma = .25;
//        final int hidden = 64;
//
//        MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
//    		.seed(Constants.NEURAL_NET_SEED)
//            .iterations(20)
//            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//            .learningRate(.1)
//            //.updater(Updater.NESTEROVS).momentum(0.9)
//            //.updater(Updater.RMSPROP).rho(conf.getRmsDecay())//.rmsDecay(conf.getRmsDecay())
//            .updater(Updater.ADAM)
//            .weightInit(WeightInit.XAVIER)
//            .regularization(true)
//            .l2(.01)
//            .list()
//            .layer(0, new DenseLayer.Builder().nIn(18).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(1, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(2, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(3, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(4, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(5, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(6, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(7, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(8, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//			.layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(hidden).nOut(1).build())
//			.pretrain(false)
//			.backprop(true)
//			.build()
//		);
        
        // ******************************************************************************************************************************************************************

//        final String bezeichner = "setting4";
//        final int trainEpochs = 100;
//        final int saveEpochs = 5000;
//        final int numberTestMatches = 500;
//        final int startEpoch = 1;
//        final double epsilon = .25;
//        final double gamma = .25;
//        final int hidden = 64;
//
//        MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
//    		.seed(Constants.NEURAL_NET_SEED)
//          .iterations(30)
//          .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//          .learningRate(.1)
//          .updater(Updater.ADAM)
//          .weightInit(WeightInit.XAVIER)
//          .regularization(true)
//          .l2(.01)
//          .list()
//          .layer(0, new DenseLayer.Builder().nIn(18).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(1, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(2, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(3, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(4, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(5, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(6, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(7, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(8, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//			.layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(hidden).nOut(1).build())
//			.pretrain(false)
//			.backprop(true)
//			.build()
//		);
        
        // ******************************************************************************************************************************************************************

//        final String bezeichner = "setting5";
//        final int trainEpochs = 100;
//        final int saveEpochs = 5000;
//        final int startEpoch = 1;
//        final int numberTestMatches = 500;
//        final double epsilon = .25;
//        final double gamma = .25;
//        final int hidden = 64;
//
//        MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
//    		.seed(Constants.NEURAL_NET_SEED)
//          .iterations(10)
//          .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//          .learningRate(.1)
//          .updater(Updater.ADAM)
//          .weightInit(WeightInit.XAVIER)
//          .regularization(true)
//          .l2(.01)
//          .list()
//          .layer(0, new DenseLayer.Builder().nIn(18).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(1, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(2, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(3, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(4, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(5, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(6, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(7, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(8, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//			.layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(hidden).nOut(1).build())
//			.pretrain(false)
//			.backprop(true)
//			.build()
//		);
        
        // ******************************************************************************************************************************************************************

//        final String bezeichner = "setting6";
//        final int trainEpochs = 100;
//        final int saveEpochs = 5000;
//        final int startEpoch = 1;
//        final int numberTestMatches = 500;
//        final double epsilon = .25;
//        final double gamma = .25;
//        final int hidden = 64;
//
//        MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
//    		.seed(Constants.NEURAL_NET_SEED)
//          .iterations(5)
//          .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//          .learningRate(.1)
//          .updater(Updater.ADAM)
//          .weightInit(WeightInit.XAVIER)
//          .regularization(true)
//          .l2(.01)
//          .list()
//          .layer(0, new DenseLayer.Builder().nIn(18).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(1, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(2, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(3, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(4, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(5, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(6, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(7, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(8, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//			.layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(hidden).nOut(1).build())
//			.pretrain(false)
//			.backprop(true)
//			.build()
//		);
        
        // ******************************************************************************************************************************************************************

//        final String bezeichner = "setting7";
//        final int trainEpochs = 100;
//        final int saveEpochs = 5000;
//        final int startEpoch = 1;
//        final int numberTestMatches = 500;
//        final double epsilon = .25;
//        final double gamma = .25;
//        final int hidden = 64;
//
//        MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
//    		.seed(Constants.NEURAL_NET_SEED)
//            .iterations(5)
//            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//            .learningRate(.1)
//            //.updater(Updater.NESTEROVS).momentum(0.9)
//            //.updater(Updater.RMSPROP).rho(conf.getRmsDecay())//.rmsDecay(conf.getRmsDecay())
//            .updater(Updater.ADAM)
//            .weightInit(WeightInit.XAVIER)
//            .regularization(true)
//            .l2(.01)
//            .list()
//            .layer(0, new DenseLayer.Builder().nIn(18).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(1, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(2, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(3, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(4, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(5, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(6, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(7, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(8, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(9, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(10, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(11, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(12, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(13, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(14, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(15, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//        	.layer(16, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//			.layer(17, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(hidden).nOut(1).build())
//			.pretrain(false)
//			.backprop(true)
//			.build()
//		);
        
        // ******************************************************************************************************************************************************************

//	      final String bezeichner = "setting1xxl2";
//	      final int startEpoch = 1;
//	      final int maxEpochs = 100000;
//	      final int numberTestMatches = 500;
//	      final int trainEpochs = 100;
//	      final int saveEpochs = 10000;
//	      final double epsilon = .25;
//	      final double gamma = .25;
//	      final int hidden = 32;
//	
//	      MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
//	  		.seed(Constants.NEURAL_NET_SEED)
//	          .iterations(15)
//	          .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//	          .learningRate(.1)
//	          .updater(Updater.ADAM)
//	          .weightInit(WeightInit.XAVIER)
//	          .regularization(true)
//	          .l2(.01)
//	          .list()
//	          .layer(0, new DenseLayer.Builder().nIn(18).nOut(hidden).activation(Activation.RELU).build())
//	      	  .layer(1, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//	      	  .layer(2, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//	      	  .layer(3, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//	      	  .layer(4, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//	      	  .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(hidden).nOut(1).build())
//	      	  .pretrain(false)
//	      	  .backprop(true)
//	      	  .build()
//		  );
      
      // ******************************************************************************************************************************************************************

//	      final String bezeichner = "setting2xxl2";
//	      final int startEpoch = 1;
//	      final int maxEpochs = 100000;
//	      final int numberTestMatches = 500;
//	      final int trainEpochs = 100;
//	      final int saveEpochs = 100;
//	      final double epsilon = .25;
//	      final double gamma = .5;
//	      final int hidden = 64;
//	
//	      MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
//    		  .seed(Constants.NEURAL_NET_SEED)
//	          .iterations(10)
//	          .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//	          .learningRate(.1)
//	          .updater(Updater.ADAM)
//	          .weightInit(WeightInit.XAVIER)
//	          .regularization(true)
//	          .l2(.01)
//	          .list()
//	          .layer(0, new DenseLayer.Builder().nIn(18).nOut(hidden).activation(Activation.RELU).build())
//	      	  .layer(1, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//	      	  .layer(2, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//	      	  .layer(3, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//	      	  .layer(4, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
//	      	  .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(hidden).nOut(1).build())
//	      	  .pretrain(false)
//	      	  .backprop(true)
//	      	  .build()
//		  );
    
    // ******************************************************************************************************************************************************************

      final String bezeichner = "setting3xxl2";
      final int trainEpochs = 100;
      final int saveEpochs = 5000;
      final int startEpoch = 1;
      final int maxEpochs = 100000;
      final int numberTestMatches = 500;
      final double epsilon = .25;
      final double gamma = .25;
      final int hidden = 64;

      MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
  		.seed(Constants.NEURAL_NET_SEED)
          .iterations(20)
          .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
          .learningRate(.1)
          .updater(Updater.ADAM)
          .weightInit(WeightInit.XAVIER)
          .regularization(true)
          .l2(.01)
          .list()
          .layer(0, new DenseLayer.Builder().nIn(18).nOut(hidden).activation(Activation.RELU).build())
      	.layer(1, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
      	.layer(2, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
      	.layer(3, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
      	.layer(4, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
      	.layer(5, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
      	.layer(6, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
      	.layer(7, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
      	.layer(8, new DenseLayer.Builder().nIn(hidden).nOut(hidden).activation(Activation.RELU).build())
			.layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(hidden).nOut(1).build())
			.pretrain(false)
			.backprop(true)
			.build()
		);
      
      // ******************************************************************************************************************************************************************

//		final String bezeichner = "setting9";
//		final int trainEpochs = 100;
//      final int startEpoch = ???;
//		final double epsilon = .25;
//		final double gamma = .25;
//		
//		MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(Model1ComputerPlayer.class.getClassLoader().getResourceAsStream("experiment/setting7.zip"));
      
      // ******************************************************************************************************************************************************************
        
        model.init();

        DQN<?> dqn = new DQN<>(model);
        
        final TTTPlayer perfectPlayer = new PerfectPlayer();

    	final Evaluator[] evaluators = new Evaluator[] {new PerfectEvaluator(), new DQNEvaluator(dqn)};

        System.out.println("done");
        
        
        // ------------------------------------------------------------------------------------------------------------------
        // DISPLAY
        
        int avg = 25;
		
		ChartDisplayDouble display = new ChartDisplayDouble(0d, maxEpochs, 0d, numberTestMatches, Color.BLACK, Color.GREEN);
		
		Map<String, Color> map = new HashMap<>();
//		map.put("setting1", Color.CYAN);
		map.put("setting1xxl", Color.CYAN);
		map.put("setting1xxl2", Color.BLUE);
		map.put("setting2xxl", Color.MAGENTA);
		map.put("setting2xxl2", Color.RED);
		map.put("setting3xxl", Color.PINK);
//		map.put("setting4", Color.PINK);
//		map.put("setting5", Color.ORANGE);
//		map.put("setting6", Color.GRAY);
//		map.put("setting7", Color.YELLOW);
		for (String key : map.keySet()) try (BufferedReader reader = new BufferedReader(new FileReader(new File("src/main/resources/experiment/" + key + ".txt")))) {
			List<PunktDoubleDouble> matchResults = new ArrayList<>();
			while (reader.ready()) {
				String line = reader.readLine();
				if (line != null && line.length()> 0) matchResults.add(new PunktDoubleDouble(null, null, Double.parseDouble(line.substring(0,  line.indexOf("|"))), Double.parseDouble(line.substring(line.indexOf("|")+1))));
			}
            display.addOrReplace(new StreckenzugDoubleDouble(key + " (avg25)", map.get(key).darker().darker(), TTTUtil.average(matchResults, 25).toArray(new PunktDoubleDouble[0])), false);
		}
		display.display();
		
		List<PunktDoubleDouble> matchResults = new ArrayList<>();
    	matchResults.add(new PunktDoubleDouble(null, null, 0d, 0d));
    	
//		List<PunktDoubleDouble> meanErrorBefore = new ArrayList<>();
//		List<PunktDoubleDouble> meanErrorAfter = new ArrayList<>();
		
        // ------------------------------------------------------------------------------------------------------------------
        // TRAINING
        
        System.out.println("Train model....");
        
        
        List<HistoryMap> history = new ArrayList<>();
        HistoryMap currentHistory = new HistoryMap(); 

        int epoch = startEpoch-1;
        while(++epoch <= maxEpochs) {
        	
        	long start = System.currentTimeMillis();
        	
        	TicTacToe ttt = new TicTacToe();
        	
        	while (!ttt.isFinal()) {

        		int current = ttt.getNext();
        		int[] available = ttt.getAvailableMoves();
        		int[] availableBinary = ttt.getAvailableMovesBinary();

        		double[] oldValues = TTTUtil.predict(dqn, ttt);
        		
        		// select next move (e-greedy, only available moves)
        		int next = random.nextDouble()<epsilon? available[random.nextInt(available.length)] : TTTUtil.maxValueIndex(oldValues, availableBinary, current);
        		int[] features = TTTUtil.features(ttt, next);
        		
        		// evaluate next move
        		TicTacToe nextTTT = TTTUtil.move(ttt, next);
        		
        		double reward = Double.NaN;
        		if (nextTTT.isFinal()) reward = nextTTT.isWin();
        		else {
        			
        			//TicTacToe nextTTT2 = move(nextTTT, maxValueIndex(predict(model, features(nextTTT)), nextTTT.getAvailableMovesBinary()));
        			TicTacToe nextTTT2 = new TicTacToe(nextTTT.getBoard());
        			perfectPlayer.draw(nextTTT2, null);
        			
        			
            		if (nextTTT2.isFinal()) reward = oldValues[next] + gamma*nextTTT2.isWin();
            		else reward = oldValues[next] + gamma*TTTUtil.maxValue(TTTUtil.predict(dqn, nextTTT2), nextTTT2.getAvailableMovesBinary(), current);
        		}
        		
        		// new value
        		if (!currentHistory.containsKey(features)) currentHistory.put(features, new ArrayList<Double>());
        		currentHistory.get2(features).add(reward);
				
		    	
				// actually make next move
				ttt = nextTTT;
        	}
        	
        	// train and evaluate
        	if (epoch >= trainEpochs && epoch%trainEpochs==0) {
        		
        		history.add(0, currentHistory);
        		while (history.size() > 10) history.remove(10);
        		currentHistory = new HistoryMap();
        		
//        		System.out.println();
//        		List<String> printList = history.printList(); 
//        		printList.forEach(each -> System.out.println(each));
//        		System.out.println();
        		

//        		double[] before = new double[history.size()];
//        		double errorBefore = 0d;
//        		for (int i = 0; i < history.size(); i++) {
//        			before[i] = evaluateToHistory(dqn, history.get(i));
//        			errorBefore += before[i];
//        		}
//        		meanErrorBefore.add(new PunktDoubleDouble(null, null, epoch, errorBefore/history.size()));
//                display.addOrReplace(new StreckenzugDoubleDouble("meanErrorBefore", Color.BLUE, meanErrorBefore.toArray(new PunktDoubleDouble[0])));
        		
        		TTTUtil.train(dqn, history);

//        		double[] after = new double[history.size()];
//        		double errorAfter = 0d;
//        		for (int i = 0; i < history.size(); i++) {
//        			after[i] = evaluateToHistory(dqn, history.get(i));
//        			errorAfter += after[i];
//        		}
//        		meanErrorAfter.add(new PunktDoubleDouble(null, null, epoch, errorAfter/history.size()));
//                display.addOrReplace(new StreckenzugDoubleDouble("meanErrorAfter", Color.MAGENTA, meanErrorAfter.toArray(new PunktDoubleDouble[0])));

        		
        		System.out.println("--------------------------------------------------------------------------------------");
            	
        		int[] results = new int[] {0, 0, 0};
        		
//        		try (PrintStream ps = new PrintStream(new OutputStream() {@Override public void write(int b) throws IOException {}})) {
//	            	for (int i = 0; i < numberTestMatches; i++) { 
//	            		int current = (results[0] + results[1] + results[2])%2;
//	            		TicTacToe t3 = TTTGame.play(players, current, null, ps); 
//	            		switch (t3.isWin()) {
//	        	    		case  0: results[2] = results[2]+1; break;
//	        	    		case  1: results[current] += 1; break;
//	        	    		case -1: results[current==0? 1 : 0] += 1;
//	        	    	}
//	            	}
//            	
//            	  matchResults.add(new PunktDoubleDouble(null, null, epoch, results[0]));
//	                
//            	}
        		
        		{	
	            	int index;
	            	List<TicTacToe> ttts; 
	            	
	            	index = -1; // perfect fängt an
	            	ttts = new ArrayList<>();
	            	ttts.add(new TicTacToe());
	            	while (evaluators[++index%2].evaluate(ttts));
	        		for (TicTacToe t3 : ttts) switch (t3.isWin()) {
	        			case  1: results[0] += 1; break;
	        			case  0: results[2] = results[2]+1; break;
	            		case -1: results[1] += 1;
	            	}
	            	
	            	index = 0; // dqn fängt an
	            	ttts = new ArrayList<>();
	            	ttts.add(new TicTacToe());
	            	while (evaluators[++index%2].evaluate(ttts));
	        		for (TicTacToe t3 : ttts) switch (t3.isWin()) {
		        		case  1: results[1] += 1; break;
		        		case  0: results[2] = results[2]+1; break;
		        		case -1: results[0] += 1;
	        		}

	            	matchResults.add(new PunktDoubleDouble(null, null, epoch, results[0]*500d/(results[0] + results[1] + results[2])));
        		}

              display.addOrReplace(new StreckenzugDoubleDouble("match (raw)", Color.WHITE.darker().darker().darker().darker(), TTTUtil.average(matchResults, 1).toArray(new PunktDoubleDouble[0])), false);
              display.addOrReplace(new StreckenzugDoubleDouble("match (avg"+avg+")", Color.WHITE, TTTUtil.average(matchResults, avg).toArray(new PunktDoubleDouble[0])), true);

              System.out.println("[" + TTTUtil.trimInt(epoch, Integer.toString(maxEpochs).length()) + "]  " + TTTUtil.trimInt(results[0], 3) + " : " + TTTUtil.trimInt(results[2], 3) + " : " + TTTUtil.trimInt(results[1], 3) + "  (" + (System.currentTimeMillis()-start)/1e3 + " secs)");
        	
        	}

        	// saving
        	if ((epoch >= saveEpochs && epoch%saveEpochs==0) || epoch == maxEpochs) {

                ModelSerializer.writeModel(model, new File("src/main/resources/experiment/" + bezeichner + "_" + epoch + ".zip"), true);
                try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("src/main/resources/experiment/" + bezeichner + ".txt")))) {
                	for (PunktDoubleDouble each : matchResults) {
                		writer.write(Double.toString(each.getX()) + "|" + Double.toString(each.getY()));
                		writer.newLine();
                	}
                }		
        	}
        }
        
        System.out.println("... done.");
    }
}