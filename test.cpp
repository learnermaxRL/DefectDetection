#include <iostream>




int firstIndex(int input[], int size, int x) {
  /* Don't write main().
     Don't read input, it is passed as function argument.
     Return output and don't print it.
     Taking input and printing output is handled automatically.
  */
        int count = 0;
    
        if (count > size-1 ){
        
        return -1;
        
    }
    else {
        
        if (input[count] == x ) return count;
        
        else {
        
        count++;
        firstIndex(input,count,x);
        
        }
        
    }

}


int main(){

int arri[] = {9 ,8, 10, 8};

std::cout << firstIndex(arri,4,8) <<" ";

return 0;

}