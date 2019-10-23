function [ jsonobj ] = mat2json( jsonfname, matfname )

    disp('running mat2json...')
    matobj = load(matfname);
    jsonobj = jsonencode(matobj);
    fileID = fopen(jsonfname, 'w');
    fprintf(fileID, '%s', jsonobj);
end

