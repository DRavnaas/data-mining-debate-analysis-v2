package debateFilter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FilterTweets {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		List<String> candidates = new ArrayList<String>(Arrays.asList("trump","cruz","rubio","kasich","carson"));
		filterTweets(candidates);
		
	}

	private static void filterTweets(List<String> candidates) {
		// TODO Auto-generated method stub
		
		Map<String,String> regexMultCandFilter = new HashMap<String,String>();
		Map<String,String> regexCurCandFilter = new HashMap<String,String>();

		String tweet1 = "If Cruz had a detailed Obamacare replacement plan, why didn't they bring it up last 4 yrs when dems say we had 0";
		
		String regexFilter1 = "(?i).cruz";
		String regexFilter2 = "(?i)^?(rubio|trump|carson|kasich)(?!.*news)";
		String regexFilter4;
 		
		for (String curCand :candidates ){
			 String regexFilter3 = "(?i)^?(";
			
			 for(String candidate: candidates){
				if (!candidate.equalsIgnoreCase(curCand))
					regexFilter3 = regexFilter3 + candidate + "|";
			}
			
			regexFilter3 = regexFilter3.substring(0, regexFilter3.length()-1);
			
			regexFilter3= regexFilter3 + ")(?!.*news)";
			
			regexFilter4 = "(?i)." + curCand;
			
			regexMultCandFilter.put(curCand, regexFilter3);
			
			System.out.println(regexFilter4);
			System.out.println(regexFilter3);
		}
		
//		Pattern p = Pattern.compile(regexFilter2);
//		Matcher m = p.matcher(tweet1);
//		
//		if (m.find())
//			System.out.println("Match");
//		else
//			System.out.println("Not match");
		
		
		//(?i).cruz
		//^(?!.*rubio)(?!.*rubio)(?!.*Obamacare)
		//^(?!.*news)(?=.*rubio)(?=.*cruz)(?=.*plan).*$ - Contains all candidates and Not news
		// (?i)^?(rubio|trump|carson|kasich)(?!.*news)
	}

}
