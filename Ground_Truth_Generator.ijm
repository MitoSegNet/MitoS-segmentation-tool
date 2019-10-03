/*
 * Aids in the generation of ground truth labels based on raw image input 
 * 
 */

run("Colors...", "foreground=black background=white selection=yellow");

raw_image = File.openDialog("Select new image or previously processed raw image to continue segmentation");

// obtaining directory path by removing filename substring from filepath
////

filepath_array = split(raw_image, File.separator); 
len = filepath_array.length; 
index = indexOf(raw_image, filepath_array[len-1]);
directory_path = substring(raw_image, 0, index); 

////

open(raw_image);

filename = filepath_array[len-1];
if(startsWith(filename, "raw_image_")){

	end_index = lengthOf(filename);
	org_title = substring(filename, 10, end_index);

	open(directory_path + File.separator + "ground_truth_" + org_title);
}

else{
	org_title = getTitle();

	rename("raw_image_" + org_title);

	run("Select None");
	run("Duplicate...", " ");
	rename("ground_truth_" + org_title);
	run("Select All");
	run("Fill", "slice");
}


selectWindow("raw_image_" + org_title);
setTool("freehand");

run("Overlay Options...", "stroke=none width=0 fill=black set");



i = 0;
while(i==0){

	selectWindow("raw_image_" + org_title);

	waitForUser("Use the freehand selection tool to draw selections around the objects you want to segment. Click OK to segment one object." 
	+ "Close Windows and click OK to exit.");

	window_list = getList("image.titles");

	if(lengthOf(window_list)==0){
		exit();
	}
	else{
	
		//run("Fill", "slice");
		run("Add Selection...");
		
		saveAs("Tiff", directory_path + File.separator + "raw_image_" + org_title);
	
		selectWindow("ground_truth_" + org_title);
		
		run("Restore Selection");
		run("Clear", "slice");

		arr = split(org_title, ".");

		saveAs("Tiff", directory_path + File.separator + "ground_truth_" + org_title);
		org_title = arr[0] + ".tif";

		print(org_title);

		
	}	 
}

run("Overlay Options...", "stroke=none width=0 fill=none set");

