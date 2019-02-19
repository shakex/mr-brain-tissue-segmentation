function val = get_variable_via_load(filepath)
if(exist(filepath,'file') == 2)
    var_struct = load(filepath);
    name_cell = fieldnames(var_struct);
    val = getfield(var_struct,char(name_cell));
elseif(exist(filepath, 'file') == 0)
    msgbox('文件不存在！','Error','Error');
end
