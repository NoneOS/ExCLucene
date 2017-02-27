import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.Hashtable;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.jsoup.Jsoup;
import org.jsoup.helper.DataUtil;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.parser.Parser;
import org.jsoup.select.Elements;

import java.io.UnsupportedEncodingException;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.Charset;
import java.nio.charset.CharsetEncoder;
import java.nio.charset.StandardCharsets;

public class Gov2Parse {
	//for gov2
	static final String documentHeadBegin_lable = "<DOC>";
	static final String documentUrl_lable = "<DOCHDR>";
	static final String documentID_lable = "<DOCNO>";
	static final String documentHeadEnd_lable = "</DOCHDR>";
	static final String documentHtmlEnd_lable = "</DOC>";
	
	static final String origin_folder_path = "/media/dataset_disk/DOTGOV2/gov2-corpus/";
	static final int begin_subfoler_number = 0;
	static final int end_subfoler_number = 0;

	static final String output_folder_path = "/media/dataset_disk/DOTGOV2/gov2-corpus-parse/";
	static final String debug_path = "/media/dataset_disk/DOTGOV2/gov2-corpus-parse/debug/";


	static final String tmp_path = "/tmp/";

	
	static final String log_filename = tmp_path + "parseDOTGOV2_log.txt";
	static final String error_filename = tmp_path + "DOTGOV2_error.txt";
	static final String check_filename = tmp_path + "DOTGOV2_toCheck_documents.txt";
	
	static final String charsetName = "UTF-8";
	static final Pattern p_html = Pattern
			.compile("(http://|https://|ftp://|www)[A-Za-z_0-9\\./=@\\+\\-\\;]*");  // in this program, we don't filter 
	static final Pattern p_stop = Pattern.compile("[.]");
	static final Pattern p_cat = Pattern.compile("[-]");
	static final Pattern p_blank = Pattern.compile("\\s+");
	
	static final Pattern p_htmlBegin = Pattern.compile(".*(<html|<HTML).*");
	static final Pattern p_htmlEnd = Pattern.compile(".*(</html>|</HTML>).*");	
	
	
	static final Pattern p_DOCNO = Pattern.compile("<DOCNO>.*(GX([0-9]|-)*).*</DOCNO>");
	

//	static Hashtable<String, Integer> ht = new Hashtable<String, Integer>();


	static FileWriter fw_check;
	static FileWriter fw_error;
	static FileWriter fw_log;
	//
	
	public static void main(String[] args) throws IOException {

		fw_log = new FileWriter(new File(log_filename));
		fw_log.write("origin_folder_path:" + origin_folder_path + "\n");
		fw_log.write("output_folder_path:" + output_folder_path + "\n");
		fw_log.flush();
//		File[] folder = new File(origin_folder_path.toString()).listFiles();
		String subfolderName = "";
		
		fw_check = new FileWriter(new File(check_filename));
	    fw_error = new FileWriter(new File(error_filename));
		
        for(int number = begin_subfoler_number; number <= end_subfoler_number; number++)
		{
			subfolderName = String.format("GX%03d", number);
			fw_log.append("process: " + subfolderName + "\n");

			File subfolder = new File(origin_folder_path + subfolderName);
			process_oneFolder(subfolder, output_folder_path);
		}
		fw_log.flush();
		
        fw_check.close();	
        fw_error.close();
        fw_log.close();
       
	}
	
	public static void process_oneFolder(File subfolder,
			String output_folder_path) throws IOException {

		if (subfolder.isDirectory()) {
			
			FileWriter fw_debug = new FileWriter(new File( debug_path + subfolder.getName() + ".txt"));	
//			FileWriter fw_debug = new FileWriter(new File( debug_path + subfolder.getName() + ".txt"));
			
			String newfolderStr = new String(output_folder_path
					+ subfolder.getName());
			File newfolder = new File(newfolderStr);
			if (newfolder != null && !newfolder.exists())
				newfolder.mkdirs();

			File[] infolder = subfolder.listFiles();
			fw_debug.write(newfolderStr + " files: " + infolder.length + "\n");
			fw_debug.flush();

			for (File inf : infolder) {
//				if(inf.getName().compareTo("30") == 0)
//				{	
					fw_log.append(inf.getName() + " ");
					String newfileName = new String(newfolderStr + "/" + inf.getName());
					process_oneFile(inf, newfileName, fw_debug);
//				}
			}
			
			System.gc();
			
//			fw_debug.close();
		}
	}

	public static void process_oneFile(File inf, String newfileName, FileWriter fw_debug) 
			throws IOException
	{
		FileWriter fw_newfile = new FileWriter(new File(newfileName));
		BufferedReader br = new BufferedReader(new FileReader(inf));

		String line = br.readLine();
		assert(line.startsWith(documentHeadBegin_lable));
		long lineno = 1;
		int documentNum = 0;
		int IDNum = 0;
		int htmlStartNum = 0;
		while ((line = br.readLine()) != null) {
			lineno++;
			boolean breakflag = false;
			boolean haveUrl = false;
			boolean haveID = false;
			String Url_str = null;
			String ID_str = null;
			String baseUrl = "";
			while (line != null) {

				//System.out.println( lineno + " line: " + line);
				if (line.startsWith(documentID_lable) == true) {
					Matcher m_DOCNO = p_DOCNO.matcher(line);
					if(m_DOCNO.find())
						ID_str = m_DOCNO.group(1);
					else
						{
							fw_error.write("DOCNO error:" + newfileName + " lineno:" +lineno + "\n");
							fw_error.flush();
							System.exit(1);
						}
					haveID = true;
//					System.out.println("ID_str:" + ID_str);
					IDNum++;
				} else if (line.startsWith(documentUrl_lable) == true) {
					baseUrl = br.readLine();
					lineno++;
					//System.out.println("lineno:" + lineno);
					Url_str = filterLineString(baseUrl);
					//System.out.println("Url_str:" + Url_str);
					haveUrl = true;
				} 
				else if (line.startsWith(documentHeadEnd_lable) == true) {
					breakflag = true;
					htmlStartNum++;
					break;
				}
				line = br.readLine();
				lineno++;
//				System.out.println("line:" + line);
			}

			//System.out.println("After break, ID_str:" + ID_str);
			
			if (haveID == true && (htmlStartNum != IDNum || breakflag != true ||  haveUrl != true ))
			{
				fw_error.append("htmlStartNum:" + htmlStartNum + "\t"
								+ "IDNum: " + IDNum + "\t"
								+ "ID_str: " + ID_str + "\t"
								+ "Url_str: " + Url_str + "\n");
				fw_error.flush();
			}
			//System.out.println( lineno + " ID_str: " + ID_str);
			if (line == null)
			{
				break;
			}

			//aSystem.out.println("ID_str : " + ID_str);
			
			StringBuilder html_sb = new StringBuilder();
			html_sb.setLength(0);
			
			while ((line = br.readLine())!= null && !line.endsWith(documentHtmlEnd_lable)) {
				lineno++;
				line = line.trim();
				if (!line.equalsIgnoreCase(""))
					html_sb.append(line + "\n");

			}		
			
			++documentNum;
			
			String htmlStr = html_sb.toString();
			StringBuilder plainTextSB ;
			
			try
			{
				
				Document doc = Jsoup.parse(htmlStr, baseUrl);
				
				HtmlParser formatter = new HtmlParser();
			   	plainTextSB = new StringBuilder(formatter.getPlainText(doc));
				
				Element body = doc.select("body").first();
				if(body != null && body.toString().compareTo("") == 0)
				{
					System.out.println("check: " + ID_str);
					fw_check.append(ID_str + "\n");
					fw_check.flush();
				}
				Elements imgElements = doc.select("img[alt]");
				for (Element imgElement:imgElements)
				{
					plainTextSB.append("\n" + imgElement.attr("alt"));
				}
			}
			catch(IllegalArgumentException ie)
			{
				fw_error.append("IllegalArgumentException error:" + ID_str + "\n");
				fw_error.flush();
		
				line = br.readLine();
				lineno++;
				continue;
			}
			String plainText = plainTextSB.toString();
			
			String[] strArr = plainText.split("\n");

			StringBuffer filteredTextBuffer = new StringBuffer();
			int resultLineNum = 0;
			for (int i = 0; i < strArr.length; i++) {
				// System.out.print("strArr[i]: " + strArr[i] + "\n");
				assert (strArr[i] != null);
				String resultLine = filterLineString(strArr[i]);
				// System.out.println("resultLine:" + "[" + resultLine + "]");
				
				if (resultLine.length() != 0) {
					resultLineNum++;
					filteredTextBuffer.append(resultLine + "\n");
				}
			}
			// System.out.print("resultLineNum: " + resultLineNum +
			// "\n");
			
			
			fw_newfile.append(ID_str + "\n");
			fw_newfile.append(Url_str + "\n");			
			fw_newfile.append(resultLineNum + "\n");
			fw_newfile.append(filteredTextBuffer.toString());

			fw_newfile.append("-------------------------------------------------------------------\n");
			fw_newfile.flush();
			
			strArr = null;

		}
//		System.exit(0);
		fw_newfile.close();
	
		br.close();

//		fw_log.append("file infomation:\n"
//				+ "htmlStartNum: " + htmlStartNum + "\n");
//		fw_log.append("IDNum: " + IDNum + "\n");
//		fw_log.append("lineno: " + lineno + "\n");
		fw_log.append("documentNum: " + documentNum + "\n");
		fw_log.flush();
		assert (IDNum == documentNum);
	
		fw_debug.write(inf.getName() + " documentNum: " + documentNum + "\n");
		fw_debug.flush();
		
	}
	
	public static String filterUrlString(String url)
	{
		String tempStr = url;
		tempStr = tempStr.replaceAll("[^\\p{ASCII}]", " "); // non-ASCII
		tempStr = tempStr.replaceAll("[~`!@#$%\\^&*\\(\\)_+=\\[\\]\\{\\}|\\\\;:'\",<>\\/?]", " ");
		tempStr = tempStr.replaceAll("[-]{1,}", " ");
		tempStr = tempStr.replaceAll("[.]{1,}", " ");
		
		String ret = "";	
		tempStr = tempStr.trim() + "\n";
		String[] rowStrArr = tempStr.split("\\s+");
		for (int j = 0; j < rowStrArr.length; j++) {
			String curStr = rowStrArr[j];
			int curStrLength = curStr.length();
			if (curStrLength == 0 || curStrLength > 300)
			{	
//				System.out.println("ID_str : " + ID_str);
//				System.out.println("rowStrArr.length: " + rowStrArr.length);
//				System.out.println("curStrLength: " + curStrLength);
//				System.out.println("curStr: " + curStr);	
				System.out.println("in url, curStrLength == 0 || curStrLength > 300");
				System.exit(1);
			}
			Matcher m_cat = p_cat.matcher(curStr);
			if (m_cat.find()) {
				if (m_cat.end() == curStr.length())  //filter begin '-' 
					curStr = curStr.substring(0,
							curStr.length() - 1);
				else if (m_cat.end() == 1)  // filter end '-' 
					curStr = curStr.substring(1,
							curStr.length() - 1);
			}
			if(curStr.isEmpty())
				continue;
			ret += curStr + " ";
		}
		ret = filterBlankSpace(ret.trim());
		return ret;
	}
	
	public static String filterLineString(String line)
	{
		
		String tempStr = line.trim();
		// filter html
//		Matcher m_html = p_html.matcher(tempStr);
//		int last_start = 0;
//		String newline = "";
//		while (m_html.find()) {
//			newline += tempStr.substring(last_start,
//					m_html.start());
//			last_start = m_html.end();
//		}
//		newline += tempStr.substring(last_start,
//				tempStr.length());
//		tempStr = newline;
		// System.out.println("tempStr(filter html):" + "[" +
		// tempStr + "]");
		
		tempStr = tempStr.replaceAll("[^\\p{ASCII}]", " "); // non-ASCII
		tempStr = tempStr.replaceAll("[~`!@#$%\\^&*\\(\\)_+=\\[\\]\\{\\}|\\\\;:'\",<>\\/?]", " ");
		tempStr = tempStr.replaceAll("[-]{1,}", " ");
		tempStr = tempStr.replaceAll("[.]{1,}", " ");
		
		String ret = "";	
		//String[] rowStrArr = line.trim().split("\\s+");
		String[] rowStrArr = tempStr.trim().split("\\s+");
		for (int j = 0; j < rowStrArr.length; j++) {
			String curStr = rowStrArr[j];
//			curStr = curStr.replaceAll("[^\\p{ASCII}]", " "); // non-ASCII
//			curStr = curStr.replaceAll("[~`!@#$%\\^&*\\(\\)_+=\\[\\]\\{\\}|\\\\;:'\",<>\\/?]", " ");
//			curStr = curStr.replaceAll("[-]{1,}", " ");
//			curStr = curStr.replaceAll("[.]{1,}", " ");
//			curStr = filterBlankSpace(curStr);	
//			curStr = curStr.trim();
			int curStrLength = curStr.length();
			if (curStrLength == 0 || curStrLength > 100)
			{	
//				System.out.print("in line, curStrLength == 0 || curStrLength > 300");
//				System.exit(1);
				continue;
			}
//			Matcher m_stop = p_stop.matcher(curStr);
//			if (m_stop.find()) {
//				if (m_stop.end() == curStr.length()) {
//					curStr = curStr.substring(0,
//							curStr.length() - 1);
//				}
//				else
//					if(m_stop.end() == 1)
//						curStr = curStr.substring(1, curStr.length());
//			}
			
//			Matcher m_cat = p_cat.matcher(curStr);
//			if (m_cat.find()) {
//				if (m_cat.end() == curStr.length())
//					curStr = curStr.substring(0,
//							curStr.length() - 1);
//				else if (m_cat.end() == 1)
//					curStr = curStr.substring(1,
//							curStr.length());
//			}
//			if(curStr.isEmpty())
//				continue;
			ret += curStr + " ";
		}
		//ret = filterBlankSpace(ret);	
		return ret.trim();
	}
	
	public static String filterBlankSpace(String originalStr) {
		return originalStr.trim().replaceAll("\\s+", " ");
	}

//	public static String filterPeriod(String originalStr) {
//		String tempStr = originalStr.trim();
//		StringBuffer sb = new StringBuffer();
//		String[] strArr = tempStr.split("\\s+");
//		for (int i = 0; i < strArr.length; i++) {
//			String curStr = strArr[i];
//			if (curStr.length() == 0)
//				continue;
//			if (curStr.charAt(curStr.length() - 1) == '.'
//					&& !ht.containsKey(curStr))
//				curStr = curStr.substring(0, curStr.length() - 1);
//			// do something with str
//			sb.append(curStr);
//		}
//		return sb.toString();
//	}
	
	 public static InputStream StringTOInputStream(String in) throws Exception{  
         
	        ByteArrayInputStream is = new ByteArrayInputStream(in.getBytes(charsetName));  
	        return is;  
	    }  
}
