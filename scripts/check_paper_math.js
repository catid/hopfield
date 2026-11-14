// Usage: NODE_PATH=/tmp/hopfield-md-validation/node_modules node scripts/check_paper_math.js
const fs = require('fs');
const katex = require('katex');
const entries=JSON.parse(fs.readFileSync('references/review/math_ast.json','utf8'));
const errors=[];
for(const [i,e] of entries.entries()) {
 try { katex.renderToString(e.tex,{displayMode:e.display,throwOnError:true,strict:'ignore'}); }
 catch(err) { errors.push({index:i,...e,error:err.message}); }
}
const report={engine:'KaTeX',version:katex.version,count:entries.length,errors};
fs.writeFileSync('references/review/math_validation.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
process.exitCode=errors.length?1:0;
