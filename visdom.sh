for i in $(ps -ax |grep visdom |awk '{print $1}')
do
id=`echo $i |awk -F"/" '{print $1}'`
kill -9  $id
done

nohup python -m visdom.server > visdom.log &
tail -f visdom.log
