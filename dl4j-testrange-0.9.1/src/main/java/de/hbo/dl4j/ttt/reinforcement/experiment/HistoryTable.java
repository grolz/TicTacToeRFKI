package de.hbo.dl4j.ttt.reinforcement.experiment;

import java.util.HashMap;

public class HistoryTable extends HashMap<int[], Double> {
	
	@Override
	public boolean containsKey(Object o) {
		for (int[] key : this.keySet()) if (toString(key).equals(toString((int[])o))) return true;
		return false;
	}
	
	public Double get2(Object o) {
		for (int[] key : this.keySet()) if (toString(key).equals(toString((int[])o))) return this.get(key);
		return 0d;
	}
	
	public void put2(int[] i, Double d) {
		for (int[] key : this.keySet()) if (toString(key).equals(toString(i))) {
			this.put(key, d);
			return;
		}
		this.put(i, d);
	}
	
	public static String toString(int[] i) {
		String erg = "[";
		if (i!=null) for (int j = 0; j < i.length; j++) erg+=(j>0?", ":"") + Integer.toString(i[j]);
		erg+="]";
		return erg;
	}
}