package de.hbo.dl4j.ttt.classifier;

import java.util.ArrayList;
import java.util.List;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

public class TTTRecordReader extends CSVRecordReader {

	@Override
    protected List<Writable> parseLine(String line){
		
		List<Writable> ret = new ArrayList<>();
		
		int index = line.indexOf("|");
		for(String s : line.substring(0, index).split(",", -1)) ret.add(new Text(s));
		//for(int i = 0; i < line.substring(index+1).length(); i++) ret.add(new Text(line.substring(index + i + 1, index + i + 2)));
		ret.add(new Text(Integer.toString(Integer.parseUnsignedInt(line.substring(index+1), 2))));
		
		
//      for(String s : line.split(delimiter, -1)) ret.add(new Text(s));
		
		return ret;
    }
}
