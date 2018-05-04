package de.hbo.dl4j.classifier;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import de.hbo.dl4j.classifier.util.RTPlotter;

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
public class MLP_2Class_Classifier {

    protected static void doIt (
        	
        int seed,
        double learningRate,
        int batchSize,
        int nEpochs,

        int numInputs,
        int numOutputs,
        int numHiddenNodes,

        final File trainingData,
        final File testData,
        final double xMin,
        final double xMax,
        final double yMin,
        final double yMax

	) throws Exception {

    
    
        // ------------------------------------------------------------------------------------------------------------------
        // MODEL SETUP
        System.out.println("Setup model....");
        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(1)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(learningRate)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes).weightInit(WeightInit.XAVIER).activation("relu").build())
            .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes).weightInit(WeightInit.XAVIER).activation("relu").build())
            .layer(2, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes).weightInit(WeightInit.XAVIER).activation("relu").build())
            .layer(3, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes).weightInit(WeightInit.XAVIER).activation("relu").build())
            .layer(4, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes).weightInit(WeightInit.XAVIER).activation("relu").build())
            .layer(5, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).weightInit(WeightInit.XAVIER).activation("softmax").weightInit(WeightInit.XAVIER).nIn(numHiddenNodes).nOut(numOutputs).build())
            .pretrain(false)
            .backprop(true)
            .build();

        
        //StatsStorage statsStorage = new InMemoryStatsStorage();
        //UIServer uiServer = UIServer.getInstance();
        //uiServer.attach(statsStorage);

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(
    		new ScoreIterationListener(10)  //Print score every 10 parameter updates
    		,new RTPlotterListener(trainingData, xMin, xMax, yMin, yMax)
    		//,new StatsListener(statsStorage)
		);

        
        
        // ------------------------------------------------------------------------------------------------------------------
        // TRAINING
        System.out.println("Train model....");
        
        //Load the training data:
        try (RecordReader rr = new CSVRecordReader()) {
	        //rr.initialize(new FileSplit(new File("src/main/resources/classification/linear_data_train.csv")));
	        rr.initialize(new FileSplit(trainingData));
	        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,0,2);
	
	        for ( int n = 0; n < nEpochs; n++) model.fit( trainIter);
        }

        
        // ------------------------------------------------------------------------------------------------------------------
        // EVALUATION
        System.out.println("Evaluate model....");

        //Load the test/evaluation data:
        try (RecordReader rrTest = new CSVRecordReader()) {
	        rrTest.initialize(new FileSplit(testData));
	        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,0,2);
	        
	        
	        Evaluation eval = new Evaluation(numOutputs);
	        while(testIter.hasNext()){
	            DataSet t = testIter.next();
	            INDArray features = t.getFeatureMatrix();
	            INDArray lables = t.getLabels();
	            INDArray predicted = model.output(features,false);
	
	            eval.eval(lables, predicted);
	
	        }

	        //Print the evaluation statistics
	        System.out.println(eval.stats());
	        System.out.println(model.score());
        }

        
        //------------------------------------------------------------------------------------
        

        /*
        //Get test data, run the test data through the network to generate predictions, and plot those predictions:
        rrTest.initialize(new FileSplit(new File("src/main/resources/classification/linear_data_eval.csv")));
        rrTest.reset();
        int nTestPoints = 500;
        testIter = new RecordReaderDataSetIterator(rrTest,nTestPoints,0,2);
        DataSet ds = testIter.next();
        INDArray testPredicted = model.output(ds.getFeatures());
        RTPlotter.plotTestData(ds.getFeatures(), ds.getLabels(), testPredicted, allXYPoints, predictionsAtXYPoints, nPointsPerAxis);
        */

        System.out.println("****************Example finished********************");
    }
    
    private static class RTPlotterListener implements TrainingListener {

        private final double xMin;
        private final double xMax;
        private final double yMin;
        private final double yMax;
        
        private final DataSet ds;
        private RTPlotter plotter = null;
        
        private int steps = 0;
        
    	public RTPlotterListener(File file, double xMin, double xMax, double yMin, double yMax) throws IOException, InterruptedException {

	        //Get all of the training data in a single array, and plot it:
	        try (RecordReader rr = new CSVRecordReader()) {
		        rr.initialize(new FileSplit(file));
		        rr.reset();
		        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,1000,0,2);
		        ds = trainIter.next();
	        }
	        
	        this.xMin = xMin;
	        this.xMax = xMax;
	        this.yMin = yMin;
	        this.yMax = yMax;
	        
    	}

		@Override public boolean invoked() {
			return false;
		}

		@Override public void invoke() {}

		@Override public void iterationDone(Model model, int iteration) {}

		@Override public void onEpochStart(Model model) {}

		@Override public void onEpochEnd(Model model) {}

		@Override public void onForwardPass(Model model, List<INDArray> activations) {}

		@Override public void onForwardPass(Model model, Map<String, INDArray> activations) {}

		@Override public void onGradientCalculation(Model model) {}

		@Override public void onBackwardPass(Model model) {
			
			if (++steps < 3) return;
			steps = 0;
			
	        //Let's evaluate the predictions at every point in the x/y input space
	        int nPointsPerAxis = 100;
	        double[][] evalPoints = new double[nPointsPerAxis*nPointsPerAxis][2];
	        int count = 0;
	        for( int i=0; i<nPointsPerAxis; i++ ){
	            for( int j=0; j<nPointsPerAxis; j++ ){
	                double x = i * (xMax-xMin)/(nPointsPerAxis-1) + xMin;
	                double y = j * (yMax-yMin)/(nPointsPerAxis-1) + yMin;

	                evalPoints[count][0] = x;
	                evalPoints[count][1] = y;

	                count++;
	            }
	        }

	        INDArray allXYPoints = Nd4j.create(evalPoints);
	        INDArray predictionsAtXYPoints = ((MultiLayerNetwork)model).output(allXYPoints);

	        if (plotter==null) plotter = RTPlotter.plotTrainingData(ds.getFeatures(), ds.getLabels(), allXYPoints, predictionsAtXYPoints, nPointsPerAxis);
	        else plotter.refresh(allXYPoints, predictionsAtXYPoints, nPointsPerAxis);
		}
		
	}
}





























