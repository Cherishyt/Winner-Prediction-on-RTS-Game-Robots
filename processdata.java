package part;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.InputStream;
import java.io.SequenceInputStream;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;

import org.jdom.Document;
import org.jdom.Element;
import org.jdom.input.SAXBuilder;
import org.jdom.output.XMLOutputter;

import rts.TraceEntry;
import util.XMLWriter;

public class processdata {
	public static void main(String[] args) {
		try {
			String all_xmldir="F:\\eclipse\\java-mars\\workspace\\microRTS\\tournament_NonMCTS\\traces";
			String random_xmldir="F:\\yutian\\Datasets\\datasets_nonMCTS\\randomsample";
		    
		    analysisxml(all_xmldir,random_xmldir );
		    	
		}catch(Exception e) {
        	e.printStackTrace();
        	}
	}
	
	public static void analysisxml(String all_dir,String random_dir){
		//String filename=null;
		String path=null;
		File allfile=new File(all_dir);
		try{
			if(allfile.isDirectory()){
				File xmlfiles=allfile;
				for(File xmlfile:xmlfiles.listFiles()){
					String xmlfilepath=xmlfile.getPath();
					String xmlfilename=xmlfile.getName();
					System.out.print(xmlfilepath);
					System.out.print("\n");
							
					InputStream is = new FileInputStream(xmlfile);
					Enumeration<InputStream> streams = Collections.enumeration(
							Arrays.asList(new InputStream[] {
									new ByteArrayInputStream("<root>".getBytes()),
									is,
									new ByteArrayInputStream("</root>".getBytes()),
									}));
					SequenceInputStream seqStream = new SequenceInputStream(streams);
					SAXBuilder saxBuilder = new SAXBuilder();
					Document document = saxBuilder.build(seqStream);
					Element rootElement=document.getRootElement();
							
					Element winner_e = rootElement.getChild("winner");
					int winner=Integer.parseInt(winner_e.getAttributeValue("winner"));
							
					Element entries_e = rootElement.getChild("rts.Trace").getChild("entries");
					List<Element> elements =  entries_e.getChildren();
					int entries_length=elements.size()-1;
					int[] cycles_array =new int[entries_length];
					for(int i=0;i<entries_length;i++) {
						Element traceentry_e=elements.get(i+1);
						cycles_array[i]=Integer.parseInt(traceentry_e.getAttributeValue("time"));
						}
							
					int[] random=new int[3];
					Random rand=new Random();
					for(int i=0;i<3;i++){
						random[i]=rand.nextInt(entries_length);
						System.out.print(cycles_array[random[i]]);
						System.out.print("\n");
						}
							
					for(int i=0;i<3;i++){
						XMLWriter xml=null;
						File folder = new File(random_dir);
						Element root=new Element("root");
						Element new_traceentry=elements.get(random[i]+1);
						//System.out.print(new_traceentry);
						root.addContent((Element)new_traceentry.clone());
						Element w=new Element("winner");
						w.setAttribute("winner", String.valueOf(winner));
						root.addContent(w);
						        
						String newfilename=xmlfilename.substring(0,xmlfilename.lastIndexOf("."))+"-s"+i+".xml";
						System.out.print(newfilename);
						System.out.print("\n");
						String newxmldir = Paths.get(random_dir, newfilename).toString();
						XMLOutputter outputter=new XMLOutputter();
						outputter.output(root,new FileOutputStream(newxmldir));   
					}
				}					
			}
				//}
			//}
		}catch(Exception e) {
				e.printStackTrace();
        }
	}
}
