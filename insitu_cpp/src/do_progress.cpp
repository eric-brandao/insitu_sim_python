#include "do_progress.h"

void progress_bar(int rc, int Nrays){
    double progress = double(rc) / double(Nrays);
    // std::cout << progress << std::endl;
    // while (progress < 1.0){
        // std::cout << progress << " %\r";
        // std::cout.flush();

        int barWidth = 70;

        std::cout << "[";
        int pos = int(barWidth * progress);
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r" <<  '\r';
        std::cout.flush();
    //     progress += 0.16;
    // }
// std::cout << std::endl;
}

// void progressbar(){
// 	std::srand(time(NULL)); //seed random
// 	for(int progress=0;progress!=100;progress+=std::rand()%20){ //increment progress randomly
// 		//Delete the line below and change for loop condition to 'progress<=100' and put something meaningful in for loop progress increment in implementation.
// 		if(progress>100) progress=100;
// 		std::cout<<"[";
// 		for(int i=0;i<100;i++)
// 			if(i<progress)
// 				std::cout<<'=';
// 			else if(i==progress)
// 				std::cout<<'>';
// 			else
// 				std::cout<<' ';
// 		std::cout<<"] "<<progress<<" %"<<'\r';
// 		std::cout.flush();
// 		std::this_thread::sleep_for(std::chrono::milliseconds(500)); //sleep
// 		//Delete this line as well in implementation
// 		if(progress==100) break;
// 	}
// 	std::cout<<std::endl;
// }