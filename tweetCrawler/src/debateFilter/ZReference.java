package debateFilter;
//*************** JUST for reference **********************
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ZReference {

	public static void main(String[] args) throws IOException {		
//		try 
//		{
//		  reader = new CSVReader(new FileReader("data/After_all_225_new.csv"));
//		  String [] nextLine;
//		  int i= 1;
//		     while ((nextLine = reader.readNext()) != null) {
//		        // nextLine[] is an array of values from the line
//		        System.out.println(nextLine[0] + nextLine[1] + "etc...");
//		        i++;
//		        if (i ==10)
//		        	break;
//		     }
//		} 
//		catch (FileNotFoundException e) 
//		{
//			System.out.println("Error in reading CSV file:  ");
//			e.printStackTrace();
//		}
		
		
		
		//(?i).cruz
		//^(?!.*rubio)(?!.*rubio)(?!.*Obamacare)
		//^(?!.*news)(?=.*rubio)(?=.*cruz)(?=.*plan).*$ - Contains all candidates and Not news
		// (?i)^?(rubio|trump|carson|kasich)(?!.*news)
		
//		
//		String text = "TruCarson could actually END poverty for African Americans  #tcot #kellyfile #ccot #pjnet #gopdebate #rednationrising";
//		String regex1 = "(?i)^?(carson)";
//		
//		Pattern p = Pattern.compile(regex1);
//		Matcher m = p.matcher(text);
//		System.out.println(!m.find());
		
	}
}
