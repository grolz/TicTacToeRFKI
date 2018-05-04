package de.hbo.dl4j.ttt.reinforcement.experiment;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

public class HistoryMap extends HashMap<int[], List<Double>> {
	
	@Override
	public boolean containsKey(Object o) {
		for (int[] key : this.keySet()) if (TTTUtil.toString(key).equals(TTTUtil.toString((int[])o))) return true;
		return false;
	}
	
	public List<Double> get2(Object o) {
		for (int[] key : this.keySet()) if (TTTUtil.toString(key).equals(TTTUtil.toString((int[])o))) return this.get(key);
		return null;
	}
	
	public void put2(int[] i, List<Double> d) {
		for (int[] key : this.keySet()) if (TTTUtil.toString(key).equals(TTTUtil.toString(i))) {
			this.put(key, d);
			return;
		}
		this.put(i, d);
	}
	
	public static String toString(List<Double> list) {
		String erg = "<";
		for (int j = 0; j < list.size(); j++) erg+=(j>0?", ":"") + list.get(j);
		erg+=">";
		return erg;
	}
	
	public List<String> printList() {
		List<String> printList = new ArrayList<>();
		for (int[] key : keySet()) printList.add(TTTUtil.toString(key) + ":  " + toString(get(key)));
		Collections.sort(printList);
		return printList;
	}
}
