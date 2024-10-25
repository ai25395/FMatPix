const temml = require('./temml.cjs');  
const args = process.argv.slice(2); // 移除前两个元素，得到实际的参数数组  
if (args.length < 1) {  
  console.log('请提供一个字符串作为参数');
  process.exit(1); // 退出程序，状态码为 1 表示错误  
}  
  
// 获取latextext
const inputString = args[0];  
const resultPath = args[1]
console.log(inputString);
// 使用 temml 渲染字符串为 MathML  
const mathML = temml.renderToString(inputString);  
// 输出结果，由python程序接收  
//console.log(mathML);
const fs = require('fs');
fs.writeFileSync(resultPath, mathML, 'utf8'); // 写入文件