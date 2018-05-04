package de.hbo.dl4j.ttt.reinforcement.experiment;

import java.awt.Color;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JFrame;

import de.hbo.diagramm.diagramm2D.DiagrammDoubleDouble;
import de.hbo.diagramm.diagramm2D.StreckenzugDoubleDouble;

public class ChartDisplayDouble {
	
	private final DiagrammDoubleDouble diagramm;
	private final List<StreckenzugDoubleDouble> charts = new ArrayList<>();

	private ChartDisplayDouble(DiagrammDoubleDouble diagramm) {this.diagramm = diagramm;}
	
	public ChartDisplayDouble(double xStart, double xEnde, double yStart, double yEnde, Color backgroundColor, Color frontColor) throws Exception {
		this(0d, xStart, xEnde, 0d, yStart, yEnde, backgroundColor, frontColor);
	}
	
	public ChartDisplayDouble(double xOrigin, double xStart, double xEnde, double yOrigin, double yStart, double yEnde, Color backgroundColor, Color frontColor) throws Exception {
		this(new DiagrammDoubleDouble (xOrigin, xStart, xEnde, yOrigin, yStart, yEnde, backgroundColor, frontColor));
	}

	
	public void display() throws Exception {
		JFrame frame = new JFrame();
		frame.setSize(1200, 800);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.getContentPane().add(diagramm.getPanel());
		frame.setVisible(true);
	}

	public void addOrReplace(StreckenzugDoubleDouble chart, boolean refresh) {
		
		for (int i = charts.size()-1; i >= 0; i--) if (charts.get(i).getBezeichnung().equals(chart.getBezeichnung())) {
			diagramm.remove(charts.get(i));
			charts.remove(i);
		}
		diagramm.add(chart);
		charts.add(chart);
		
		if (refresh) diagramm.refresh();
	}
}