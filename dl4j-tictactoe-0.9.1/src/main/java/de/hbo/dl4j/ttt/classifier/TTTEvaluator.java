package de.hbo.dl4j.ttt.classifier;

import java.io.File;

import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;

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
public class TTTEvaluator {


    public static void main(String[] args) throws Exception {

    
    
        // ------------------------------------------------------------------------------------------------------------------
        // MODEL SETUP
        System.out.print("Setup model:");

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File("src/main/resources/classification/ttt_model_512.zip"));

        System.out.println("done");
        
	        	
    	try (TTTRecordReader rr2 = new TTTRecordReader()) {
	        rr2.initialize(new FileSplit(new File("src/main/resources/classification/tictactoe.txt")));
	        RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(rr2,1,10,512);

	        
	        while(iterator.hasNext()) {
	        	
	        	DataSet t = iterator.next();
	        	INDArray features = t.getFeatures();
	        	INDArray labels = t.getLabels();
	        	
	            double[] data = model.output(features,false).data().asDouble();
	            
	            int ones = 0;
	            for (int i = 0; i < data.length; i++) if (data[i] >= .35) ones++;
	            
	            if (ones != 1) System.out.println(features + " | " + labels +" : " + ones);
	        }   
	        

//            INDArray predicted = model.output(new NDArray(new float[] {1f, 0f, -1f, 0f, 0f, 0f, 0f, 0f, 0f, 1f}),false);
//            System.out.print("");
	        
//	        Evaluation eval = model.evaluate(iterator);
//	        System.out.println(new SimpleDateFormat("HH:mm:ss").format(new Date()) + " - precision: " + trimDouble(eval.precision(), 12) + ", accuracy: " + trimDouble(eval.accuracy(), 12) + ", score: " + model.score());
    	}
        
        System.out.println("... done.");
    }
    
    private static String trimDouble(double d, int length) {
    	String erg = Double.toString(d);
    	if (erg.length() > length) erg = erg.substring(0, length);
    	while (erg.length() < length) erg += (erg.indexOf(".") < 0)? "." : "0";
    	return erg;
    }
    
    private static String trimInt(int i, int length) {
    	String erg = Integer.toString(i);
    	while (erg.length() < length) erg = " " + erg;
    	return erg;
    }
}