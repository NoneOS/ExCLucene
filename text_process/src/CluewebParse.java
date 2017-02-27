import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;

import java.util.Hashtable;

public class CluewebParse {

	static final String documentHead_lable = "WARC/0.18";
	static final String documentUrl_lable = "WARC-Target-URI: ";
	static final String documentID_lable = "WARC-TREC-ID: ";

	static final String origin_folder_path = "/dataset/clueweb09/";
	static final String output_folder_path = "/dataset/clueweb09.parse/";
	static final String debug_path = "/dataset/clueweb09.parse/debug/";
	static final String error_filename = origin_folder_path + "error.txt";
	
	static final int begin_subfoler_number = 0;
	static final int end_subfoler_number = 0;
	
	static final Pattern p_html = Pattern
			.compile("(http://|https://|ftp://|www)[A-Za-z0-9_\\./=@\\+\\-\\;]*");
	static final Pattern p_stop = Pattern.compile("[.]");
	static final Pattern p_cat = Pattern.compile("[-]");
	static final Pattern p_blank = Pattern.compile("\\s+");

	static Hashtable<String, Integer> ht = new Hashtable<String, Integer>();

	public static void main(String[] args) throws IOException {

//		File[] folder = new File(origin_folder_path.toString()).listFiles();
		//subfolderName = "en0012";
		//test();
		//process_oneFolder(new File(origin_folder_path + "singleDocument"), output_folder_path);
		
		String subfolderName = "";
        for(int number = begin_subfoler_number; number <= end_subfoler_number; number++)
		{
			subfolderName = String.format("en%04d", number);
			File subfolder = new File(origin_folder_path + subfolderName);
			process_oneFolder(subfolder, output_folder_path);
		}
	    

	}
	
	public static void process_oneFolder(File subfolder,
			String output_folder_path) throws IOException {

		System.out.println("subfolder.getName(): " + subfolder.getName());	
		if (subfolder.isDirectory()) {
			
			FileWriter fw_debug = new FileWriter(new File( debug_path + subfolder.getName() + ".txt"));		
			String newfolderStr = new String(output_folder_path
					+ subfolder.getName());
			File newfolder = new File(newfolderStr);
			if (newfolder != null && !newfolder.exists())
				newfolder.mkdirs();

			File[] infolder = subfolder.listFiles();
			fw_debug.write(newfolderStr + " files: " + infolder.length + "\n");
			fw_debug.flush();

			for (File inf : infolder) {
				System.out.println(subfolder.getName() + " " + inf.getName());
				String newfileName = new String(newfolderStr + "/" + inf.getName());
				process_oneFile(inf, newfileName, fw_debug);
			}
			
			System.gc();
			
			fw_debug.close();
		}
	}

	public static void process_oneFile(File inf, String newfileName, FileWriter fw_debug) 
			throws IOException
	{
		FileWriter fw_newfile = new FileWriter(new File(newfileName));
		BufferedReader br = new BufferedReader(new FileReader(inf));

		String line = br.readLine();
		line = br.readLine();// jump first line
		long lineno = 2;
		int documentNum = 0;
		int IDNum = 0;
		int htmlStartNum = 0;
		while (line != null) {
			boolean breakflag = false;
			boolean haveUrl = false;
			boolean haveID = false;
			int content_length_lable_num = 0;
			String Url_str = null;
			String ID_str = null;
			while (line != null) {
				if (line.startsWith(documentUrl_lable) == true) {
					Url_str = line
							.substring(documentUrl_lable.length());
					haveUrl = true;
				} else if (line.startsWith(documentID_lable) == true) {
					ID_str = line.substring(documentID_lable.length());
					haveID = true;
					IDNum++;
				} 
//				else if (haveID && line.trim().startsWith("<") == true) {
//					breakflag = true;
//					htmlStartNum++;
//					break;
//				}
				else if (haveID && line.startsWith("Content-Length: ") == true) {
					content_length_lable_num++;
					if (content_length_lable_num == 2) {
						htmlStartNum++;
						breakflag = true;
						break;
					}
				}
				line = br.readLine();
				lineno++;
			}

			if (htmlStartNum != IDNum)
			{
				System.out.print("htmlStartNum: " + htmlStartNum + "\n");
				System.out.print("IDNum: " + IDNum + "\n");
				System.out.print("ID_str: " + ID_str + "\n");
				System.out.print("Url_str: " + Url_str + "\n");
				System.exit(1);
			}
			assert(content_length_lable_num == 2);
			assert (breakflag == true && haveUrl == true && haveID == true);
			assert (line != null);

			//aSystem.out.println("ID_str : " + ID_str);
			
			StringBuilder html_sb = new StringBuilder();
			html_sb.setLength(0);
			line = br.readLine();
			lineno++;

			boolean startFlag = false;
			while (line != null && !line.startsWith(documentHead_lable)) {
				if (startFlag == false && line.indexOf('<') > 0) {
					startFlag = true;
					int startIndex = line.indexOf('<');
					line = line.substring(startIndex);
				}
				line = line.trim();
				if (!line.equalsIgnoreCase(""))
				{
					startFlag = true;
					html_sb.append(line + "\n");
				}
				line = br.readLine();
				lineno++;
			}
			
			++documentNum;
			
//			if(ID_str.equalsIgnoreCase("clueweb09-en0015-35-14631"))
//			{
//				System.out.println(html_sb.toString());
//			}
			String plainText = "";
			try
			{
				Document doc = Jsoup.parse(html_sb.toString(),"http://example.com/");
				// Element title = doc.select("title").first();
				// String titleStr = "\n";
				// if (title != null)
				// titleStr = title.text();
				// titleStr.replaceAll(" +[-] +", " ");
				// fw_newfile.write(titleStr + "\n");
				// fw_newfile.flush();

				// System.out.println("TITLE : " + titleStr);

				HtmlParser formatter = new HtmlParser();
				plainText = formatter.getPlainText(doc);

				Element title = doc.select("title").first();
				if (title == null) {
					plainText = "title-title\n" + plainText;
					//System.out.println("plainText : " + plainText);
				}
			}
			catch(IllegalArgumentException ie)
			{
				FileWriter fw_error = new FileWriter(new File(error_filename), true);
				fw_error.append(ID_str + "\n");
				fw_error.flush();
				fw_error.close();
				
				line = br.readLine();
				lineno++;
				continue;
			}
			fw_newfile.write(ID_str + "\n");
			fw_newfile.flush();
			fw_newfile.write(Url_str + "\n");
			fw_newfile.flush();
			// else
			// {
			// System.out.println("ID_str : " + ID_str);
			// System.out.println("TITLE : " + title.text());
			// }
			plainText = plainText.replaceAll("[^\\p{ASCII}]", ""); // non-ASCII
			plainText = plainText.replaceAll("[~`!@#$%\\^&*\\(\\)_+=\\[\\]\\{\\}|\\\\;:'\",<>\\/?]", " ");
			plainText = plainText.replaceAll("[-]{2,}", " ");
			plainText = plainText.replaceAll("[.]{2,}", " ");

			String[] strArr = plainText.split("\n");

			StringBuffer filteredTextBuffer = new StringBuffer();
			int resultLineNum = 0;
			for (int i = 0; i < strArr.length; i++) {
				// System.out.print("strArr[i]: " + strArr[i] + "\n");
				assert (strArr[i] != null);
				String tempStr = filterBlankSpace(strArr[i]);
				// System.out.println("tempStr:" + "[" + tempStr + "]");
				assert (tempStr != null);
				if (tempStr.length() == 0)
					continue;
				// filter html
				Matcher m_html = p_html.matcher(tempStr);
				int last_start = 0;
				String newline = "";
				while (m_html.find()) {
					newline += tempStr.substring(last_start,
							m_html.start());
					last_start = m_html.end();
				}
				newline += tempStr.substring(last_start,
						tempStr.length());
				tempStr = newline;
				// System.out.println("tempStr(filter html):" + "[" +
				// tempStr + "]");

				String resultStr = "";
				tempStr = tempStr.trim() + "\n";
				String[] rowStrArr = tempStr.split("\\s+");
				for (int j = 0; j < rowStrArr.length; j++) {
					String curStr = rowStrArr[j];
					int curStrLength = curStr.length();
					if (curStrLength == 0 || curStrLength > 300)
					{
//						System.out.println("ID_str : " + ID_str);
//						System.out.println("rowStrArr.length: " + rowStrArr.length);
//						System.out.println("curStrLength: " + curStrLength);
//						System.out.println("curStr: " + curStr);	
						continue;
					}
					Matcher m_stop = p_stop.matcher(curStr);
					if (m_stop.find()) {
						if (m_stop.end() == curStr.length()) {
							curStr = curStr.substring(0,
									curStr.length() - 1);
						}
					}
					Matcher m_cat = p_cat.matcher(curStr);
					if (m_cat.find()) {
						if (m_cat.end() == curStr.length())
							curStr = curStr.substring(0,
									curStr.length() - 1);
						else if (m_cat.end() == 1)
							curStr = curStr.substring(1,
									curStr.length());
					}
					resultStr += curStr + " ";
				}
				resultStr = filterBlankSpace(resultStr.trim());

				// System.out.println("resultStr:" + "[" + resultStr +
				// "]");
				assert (resultStr != null);
				if (resultStr.length() != 0) {
					resultLineNum++;
					filteredTextBuffer.append(resultStr + "\n");
				}
				tempStr = null;
				resultStr = null;
			}
			// System.out.print("resultLineNum: " + resultLineNum +
			// "\n");

			fw_newfile.write(resultLineNum + "\n");
			fw_newfile.flush();

			fw_newfile.write(filteredTextBuffer.toString());
			fw_newfile.flush();
			fw_newfile.write("-------------------------------------------------------------------\n");
			fw_newfile.flush();


			line = br.readLine();
			lineno++;
			strArr = null;

		}
//		System.exit(0);
		fw_newfile.close();
		br.close();

		System.out.print("htmlStartNum: " + htmlStartNum + "\n");
		System.out.print("IDNum: " + IDNum + "\n");
		System.out.print("lineno: " + lineno + "\n");
		System.out.print("documentNum: " + documentNum + "\n");
		assert (IDNum == documentNum);
		
		fw_debug.write(inf.getName() + ": documentNum: " + documentNum + "\n");
		fw_debug.flush();
		
	}

	public static String filterBlankSpace(String originalStr) {
		String tempStr = originalStr.trim();
		Matcher tempStr_m = p_blank.matcher(tempStr);
		if (tempStr_m.find())
			if (tempStr_m.end() == tempStr.length()) {
				tempStr += "\n";
				tempStr = tempStr.replaceAll("\\s*\n", "");
				// System.out.println("tempStr:" + "[" + tempStr + "]");
			} else {
				tempStr = tempStr.replaceAll("\\s+", " ");
				tempStr += "\n";
				tempStr = tempStr.replaceAll("\\s*\n", "");

			}
		return tempStr;
	}

	public static String filterPeriod(String originalStr) {
		String tempStr = originalStr.trim();
		StringBuffer sb = new StringBuffer();
		String[] strArr = tempStr.split("\\s+");
		for (int i = 0; i < strArr.length; i++) {
			String curStr = strArr[i];
			if (curStr.length() == 0)
				continue;
			if (curStr.charAt(curStr.length() - 1) == '.'
					&& !ht.containsKey(curStr))
				curStr = curStr.substring(0, curStr.length() - 1);
			// do something with str
			sb.append(curStr);
		}
		return sb.toString();
	}

	public static void test() throws IOException
	{
			File inf = new File("/clueweb09-en0057-46-11858.txt");
			StringBuilder html_sb = new StringBuilder();
			BufferedReader br = new BufferedReader(new FileReader(inf));
			String line = br.readLine();
			long lineno = 2;
			int documentNum = 0;
			int IDNum = 0;
			int htmlStartNum = 0;
			
			while (line != null) {
				boolean breakflag = false;
				boolean haveUrl = false;
				boolean haveID = false;
				int content_length_num = 0;
				String Url_str = null;
				String ID_str = null;
				while (line != null) {
					if (line.startsWith(documentUrl_lable) == true) {
						Url_str = line
								.substring(documentUrl_lable.length());
						haveUrl = true;
					} else if (line.startsWith(documentID_lable) == true) {
						ID_str = line.substring(documentID_lable.length());
						haveID = true;
						IDNum++;
						// System.out.println(lineno + "::::" + line);
						if (htmlStartNum + 1 < IDNum) {
							System.out.println(lineno + "::::" + line);
							System.exit(0);
						}
					} else if (haveID && line.trim().startsWith("<") == true) {
						breakflag = true;
						htmlStartNum++;
						break;
					} else if (haveID && line.startsWith("Content-Length: ") == true) {
						content_length_num++;
						if (content_length_num == 2) {
							htmlStartNum++;
							breakflag = true;
							break;
						}
					}
					System.out.print("line: " + line + "\n");
					line = br.readLine();
					lineno++;
					
				}
				if (htmlStartNum != IDNum)
				{
					System.out.print("htmlStartNum: " + htmlStartNum + "\n");
					System.out.print("IDNum: " + IDNum + "\n");
					System.out.print("ID_str: " + ID_str + "\n");
					System.out.print("Url_str: " + Url_str + "\n");
					System.exit(1);
				}
				assert (breakflag == true && haveUrl == true && haveID == true);
				assert (line != null);
				
				html_sb.setLength(0);
				line = br.readLine();
				lineno++;

				boolean startFlag = false;
				while (line != null && !line.startsWith(documentHead_lable)) {
					if (startFlag == false && line.indexOf('<') > 0) {
						startFlag = true;
						int startIndex = line.indexOf('<');
						line = line.substring(startIndex);
					}
					line = line.trim();
					if (!line.equalsIgnoreCase(""))
						html_sb.append(line + "\n");
					line = br.readLine();
					lineno++;
				}
//				System.out.println("11111111111111111111111");
				try{
				Document doc = Jsoup.parse(html_sb.toString(),"http://example.com/");
//				System.out.println("22222222222222222222222");

				HtmlParser formatter = new HtmlParser();
				String plainText = formatter.getPlainText(doc);

				Element title = doc.select("title").first();
				if (title == null) {
					plainText = "title-title\n" + plainText;
					//System.out.println("plainText : " + plainText);
				}

				plainText = plainText.replaceAll("[^\\p{ASCII}]", ""); // non-ASCII
				plainText = plainText.replaceAll("[~`!@#$%\\^&*\\(\\)_+=\\[\\]\\{\\}|\\\\;:'\",<>\\/?]", " ");
				plainText = plainText.replaceAll("[-]{2,}", " ");
				plainText = plainText.replaceAll("[.]{2,}", " ");

				String[] strArr = plainText.split("\n");

				StringBuffer filteredTextBuffer = new StringBuffer();
				//int resultLineNum = 0;
				for (int i = 0; i < strArr.length; i++) {
					// System.out.print("strArr[i]: " + strArr[i] + "\n");
					assert (strArr[i] != null);
					String tempStr = filterBlankSpace(strArr[i]);
					// System.out.println("tempStr:" + "[" + tempStr + "]");
					assert (tempStr != null);
					if (tempStr.length() == 0)
						continue;
					// filter html
					Matcher m_html = p_html.matcher(tempStr);
					int last_start = 0;
					String newline = "";
					while (m_html.find()) {
						newline += tempStr.substring(last_start,
								m_html.start());
						last_start = m_html.end();
					}
					newline += tempStr.substring(last_start,
							tempStr.length());
					tempStr = newline;
					// System.out.println("tempStr(filter html):" + "[" +
					// tempStr + "]");

					String resultStr = "";
					tempStr = tempStr.trim() + "\n";
					String[] rowStrArr = tempStr.split("\\s+");
					for (int j = 0; j < rowStrArr.length; j++) {
						String curStr = rowStrArr[j];
						int curStrLength = curStr.length();
						if (curStrLength == 0 || curStrLength > 300)
						{
//							System.out.println("ID_str : " + ID_str);
//							System.out.println("rowStrArr.length: " + rowStrArr.length);
//							System.out.println("curStrLength: " + curStrLength);
//							System.out.println("curStr: " + curStr);	
							continue;
						}
						Matcher m_stop = p_stop.matcher(curStr);
						if (m_stop.find()) {
							if (m_stop.end() == curStr.length()) {
								curStr = curStr.substring(0,
										curStr.length() - 1);
							}
						}
						Matcher m_cat = p_cat.matcher(curStr);
						if (m_cat.find()) {
							if (m_cat.end() == curStr.length())
								curStr = curStr.substring(0,
										curStr.length() - 1);
							else if (m_cat.end() == 1)
								curStr = curStr.substring(1,
										curStr.length() - 1);
						}
						resultStr += curStr + " ";
					}
					resultStr = filterBlankSpace(resultStr.trim());
					
					//System.out.println(resultStr);
					
					// System.out.println("resultStr:" + "[" + resultStr +
					// "]");
					assert (resultStr != null);
					if (resultStr.length() != 0) {
						//resultLineNum++;
						filteredTextBuffer.append(resultStr + "\n");
					}
					tempStr = null;
					resultStr = null;
				}
				// System.out.print("resultLineNum: " + resultLineNum +
				// "\n");
				++documentNum;

				line = br.readLine();
				lineno++;
				strArr = null;		
				} catch (IllegalArgumentException e)
				{
					System.out.println(ID_str);
				}
			}
		}
}
