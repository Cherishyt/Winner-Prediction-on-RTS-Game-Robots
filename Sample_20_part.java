package part;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
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

public class Sample_20_part {
	public static void main(String[] args) {
		try {
			String all_xmldir="F:\\yutian\\Datasets\\tournament_Intercepted_part1_8";
			String random_xmldir="F:\\yutian\\Datasets\\InterceptedDatasets\\sample_part_20";
		    analysisxml(all_xmldir,random_xmldir );
		    	
		}catch(Exception e) {
        	e.printStackTrace();
        	}
	}
	
	public static void analysisxml(String all_dir,String random_dir){
		String path=null;
		File allfile=new File(all_dir);
		try{
			if(allfile.isDirectory()){
				for(File partfiles:allfile.listFiles()){
					if(partfiles.isDirectory()){
						String tracepath=Paths.get(partfiles.getPath(), "traces").toString();
						File xmlfiles=new File(tracepath);
						for(File xmlfile:xmlfiles.listFiles()){
							String xmlfilepath=xmlfile.getPath();
							String xmlfilename=xmlfile.getName();
							System.out.print(xmlfilepath);
							System.out.print("\n");

							System.out.print("Start read xml...");
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
							System.out.println("entries_length:"+entries_length+"\n");
							//create folder
							int m=0;
							for(int i=1;i<=20;i++){
								String folder=random_dir+"/samplepart"+i;
								File folderpath=new File(folder);
								if(!folderpath.exists()){
									folderpath.mkdir();
									System.out.println("Create new folder:"+folder+"\n");
								}
								switch(i){
								case 1:
									m=(int)(entries_length*0.025);
									System.out.println("sample part1: 2.5% of GameLength is"+m+"\n");
									//sample(l_0,l_1,folder,elements,winner,xmlfilename);
									break;
								case 2:
									m=(int)(entries_length*0.075);
									System.out.println("sample part2: 7.5% of GameLength is"+m+"\n");
									//sample(l_1,l_2,folder,elements,winner,xmlfilename);
									break;
								case 3:
									m=(int)(entries_length*0.125);
									System.out.println("sample part3: 12.5% of GameLength is"+m+"\n");
									//sample(l_2,l_3,folder,elements,winner,xmlfilename);
									break;
								case 4:
									m=(int)(entries_length*0.175);
									System.out.println("sample part4: 17.5% of GameLength is"+m+"\n");
									//sample(l_3,l_4,folder,elements,winner,xmlfilename);
									break;
								case 5:
									m=(int)(entries_length*0.225);
									System.out.println("sample part5: 22.5% of GameLength is"+m+"\n");
									//sample(l_4,l_5,folder,elements,winner,xmlfilename);
									break;
								case 6:
									m=(int)(entries_length*0.275);
									System.out.println("sample part6: 27.5% of GameLength is"+m+"\n");
									//sample(l_5,l_6,folder,elements,winner,xmlfilename);
									break;
								case 7:
									m=(int)(entries_length*0.325);
									System.out.println("sample part7: 32.5% of GameLength is"+m+"\n");
									//sample(l_6,l_7,folder,elements,winner,xmlfilename);
									break;
								case 8:
									m=(int)(entries_length*0.375);
									System.out.println("sample part8: 37.5% of GameLength is"+m+"\n");
									//sample(l_7,l_8,folder,elements,winner,xmlfilename);
									break;
								case 9:
									m=(int)(entries_length*0.425);
									System.out.println("sample part9: 42.5% of GameLength is"+m+"\n");
									//sample(l_8,l_9,folder,elements,winner,xmlfilename);
									break;
								case 10:
									m=(int)(entries_length*0.475);
									System.out.println("sample part10: 47.5% of GameLength is"+m+"\n");
									//sample(l_9,l_10,folder,elements,winner,xmlfilename);
									break;
								case 11:
									m=(int)(entries_length*0.525);
									System.out.println("sample part11: 52.5% of GameLength is"+m+"\n");
									//sample(l_10,l_11,folder,elements,winner,xmlfilename);
									break;
								case 12:
									m=(int)(entries_length*0.575);
									System.out.println("sample part12: 57.5% of GameLength is"+m+"\n");
									//sample(l_11,l_12,folder,elements,winner,xmlfilename);
									break;
								case 13:
									m=(int)(entries_length*0.625);
									System.out.println("sample part13: 62.5% of GameLength is"+m+"\n");
									//sample(l_12,l_13,folder,elements,winner,xmlfilename);break;
								case 14:
									m=(int)(entries_length*0.675);
									System.out.println("sample part14: 67.5% of GameLength is"+m+"\n");
									//sample(l_13,l_14,folder,elements,winner,xmlfilename);
									break;
								case 15:
									m=(int)(entries_length*0.725);
									System.out.println("sample part15: 72.5% of GameLength is"+m+"\n");
									//sample(l_14,l_15,folder,elements,winner,xmlfilename);
									break;
								case 16:
									m=(int)(entries_length*0.775);
									System.out.println("sample part16: 77.5% of GameLength is"+m+"\n");
									//sample(l_15,l_16,folder,elements,winner,xmlfilename);
									break;
								case 17:
									m=(int)(entries_length*0.825);
									System.out.println("sample part17: 82.5% of GameLength is"+m+"\n");
									//sample(l_16,l_17,folder,elements,winner,xmlfilename);
									break;
								case 18:
									m=(int)(entries_length*0.875);
									System.out.println("sample part1: 87.5% of GameLength is"+m+"\n");
									//sample(l_17,l_18,folder,elements,winner,xmlfilename);
									break;
								case 19:
									m=(int)(entries_length*0.925);
									System.out.println("sample part19: 92.5% of GameLength is"+m+"\n");
									//sample(l_18,l_19,folder,elements,winner,xmlfilename);
									break;
								case 20:
									m=(int)(entries_length*0.975);
									System.out.println("sample part20: 97.5% of GameLength is"+m+"\n");
									//sample(l_19,l_20,folder,elements,winner,xmlfilename);
									break;
								}
								XMLWriter xml=null;
								Element root=new Element("root");
								Element new_traceentry=elements.get(m+1);
								//System.out.print(new_traceentry);
								root.addContent((Element)new_traceentry.clone());
								Element w=new Element("winner");
						        w.setAttribute("winner", String.valueOf(winner));
						        root.addContent(w);
						        
						        //String newfilename=xmlfilename.substring(0,xmlfilename.lastIndexOf("."))+"-s"+i+".xml";
						        String newfilename=xmlfilename.substring(0,xmlfilename.lastIndexOf("."))+"-s0"+".xml";
						        System.out.print(newfilename);
								System.out.print("\n");
								String newxmldir = Paths.get(folder, newfilename).toString();
						        XMLOutputter outputter=new XMLOutputter();
						        outputter.output(root,new FileOutputStream(newxmldir));  
							}
							
						}
					}
				}
			}
		}catch(Exception e) {
				e.printStackTrace();
        }
	}
}
