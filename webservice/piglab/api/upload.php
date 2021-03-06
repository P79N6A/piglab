<?php
//上传文件
//echo json_encode($_COOKIE);return;
// https://www.yanjingang.com/lab/api/upload.php
$PATH = "/home/work/odp/webroot/yanjingang/lab/";
$URL = "https://www.yanjingang.com/lab/";
$SUFFIXS = ['.png','.jpg','.jpeg'];

//接收到的文件信息 
//file_put_contents($PATH.'tmp/upload.txt', var_export($_FILES,true));
$uploaded_file=$_FILES['data']['tmp_name'];  
$real_name=$_FILES['data']['name']; 
 
$result = array('status'=>0,'msg'=>'succ','data'=>array());
if(!is_uploaded_file($uploaded_file)) {  
    $result['status'] = 1;
    $result['msg'] = 'upload fail!'; 
}else{
    //保存目录
    $save_path=$PATH .'upload/'. date('ymd')."/";  
    if(!file_exists($save_path)) {  
        mkdir($save_path);  
    }
    $suffix = substr($real_name,strrpos($real_name,"."));
    if(!in_array($suffix, $SUFFIXS)){
        $result['status'] = 2;
        $result['msg'] = 'invalid ['.$suffix.'] suffix!'; 
    }else{
        $save_file=$save_path.time().rand(1,1000).$suffix;
        //file_put_contents($PATH.'upload/save_file.txt', $save_file);
        if(!move_uploaded_file($uploaded_file,iconv("utf-8","gb2312",$save_file))) {  
	    $result['status'] = 3;
            $result['msg'] = 'rename fail!' ; 
        }
	#py api_paddle	
	$url = "http://www.yanjingang.com:8020/piglab/image/digit?img_file=" . $save_file;
	$res = file_get_contents($url);
	$res = json_decode($res,true);
	if($res['code']!=0){
	    $result['status'] = 4;
            $result['msg'] = 'infer fail!'. $res['code']. ' '. $res['msg'] ;
	}else{
	    $result['data'] = $res['data'];
  	    $result['data']['requrl'] = $url;
	    $result['data']['url'] = str_replace($PATH, $URL, $save_file);
	}
    }
}

header("Content-type: application/json");
echo json_encode($result);
