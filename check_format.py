import re
def preprocess(s:str):
    """remove $$ in one line"""
    lines = s.split('\n')
    new_lines = []
    while len(lines):
        line = lines[0]
        i =  line.find('$$')
        if i != -1:
            if line[:i] != '':
                new_lines.append(line[:i])
            new_lines.append('$$')
            if line[i+2:] != '':
                lines[0] = line[i+2:]
            else:
                lines = lines[1:]
        else:
            new_lines.append(line)
            lines = lines[1:]
    return new_lines

def process(s:str):
    """
    2.行间公式前后必须有空行
    """
    lines = preprocess(s)
    # print(lines)
    inline_formula = False
    last_inline = -1
    add_newlines = 0
    new_lines = lines
    for i,line in enumerate(lines):
        if line == '$$':
            inline_formula = not inline_formula
            if not inline_formula: # recently quit
                if last_inline>0 and lines[last_inline-1] != '':
                    new_lines = new_lines[:last_inline+add_newlines]+['']+new_lines[last_inline+add_newlines:]
                    add_newlines += 1
                if i<len(lines)-1 and lines[i+1] != '':
                    new_lines = new_lines[:i+add_newlines+1]+['']+new_lines[i+1+add_newlines:]
                    add_newlines += 1
            else:
                last_inline = i
    return '\n'.join(new_lines) 

def format(s:str):
    """
    1.每一个$$前后必须有空格
    2.行间公式前后必须有空行
    """
    s = process(s)
    lines = s.split('\n')
    new_lines = []
    for i,line in enumerate(lines):            
        # normal lines
        begin_index = -1
        in_envionment = False
        while True:
            index = line.find('$',begin_index+1)
            if index == -1:
                break
            in_envionment = not in_envionment
            if not in_envionment:
                add1 = add2 = False
                remove1 = remove2 = 0
                if begin_index > 0 and line[begin_index-1]!=' ':
                    add1 = True
                if begin_index < len(line)-1 and line[begin_index+1]==' ':
                    remove1 = len(re.findall(r'^([\s]+)[^\s].*$',line[begin_index+1:])[0])
                if index < len(line)-1 and line[index+1]!=' ':
                    add2 = True
                if index > 0 and line[index-1]==' ':
                    remove2 = len(re.findall(r'^.*[^\s]([\s]+)$',line[:index])[0])
                line = line[:begin_index]+(' ' if add1 else '')+'$'+line[begin_index+remove1+1:index-remove2]+'$'+(' ' if add2 else '') + line[index+1:]
                begin_index = index + int(add1)
            else:
                begin_index = index
        new_lines.append(line)

            
    return '\n'.join(new_lines)

README = """
# For Developers

Please edit on **source.md**. After that, you should run `check_format.py` and push. However, Github formula renderer may not work properly, so you should manually change some format in order to let it work (or add an issue for not working formulas). Finally, after you have done all of this, please **replace source.md with the new README contents.**

## Note on good writing habits (in order our formater to work):

1. Don't put `$$` formulas in environments (e.g. quoting/enumerating)
2. Make sure that you use `\*` instead of `*` in formulas
3. Bold or Italic contents shouldn't start or end with formulas (i.e. `$`)
"""

if __name__ == '__main__':
    IN = './source.md'#'./test.md'
    OUT = './README.md'#'./test_out.md'
    s = open(IN,'r').read()
    # open('./test_out.md','w').write(preprocess(s))
    f = open(OUT,'w')
    f.write(README)
    f.write(format(s.replace('^*','^\*')))