!
! To compile on Alpha machine:
!    f90 -free -cpp -convert big_endian read-ter.F
! To complie on Sun:
!    f90 -ansi -free -DRECLENBYTE read-ter.F
! To complie on Linux:
!    pgf90 -Mfree -DRECLENBYTE -byteswapio read-ter.F
!
!
! To run it, type 'a.out'
!
!
program read_soil

! Read 30 sec data from a direct access data file 
!
!      The 30 sec soil data starts at 
!          latitude 90. and longitude -180.
!
!    - the file is in direct access format
!    - each direct access record contains data in one latitude circle
!      beginning at -180 degree longitude, and end at +180 degree
!    - the data is arranged to start at northernmost latitude (north pole),
!      and end at south pole

 implicit none

! declare variables

 character (len=25) :: soilfile
 character (len=1 ), allocatable, dimension(:) :: soil_char

 integer :: iunit = 10
 integer :: iunitin = 20
 integer :: rec_len_lat = 360*120
 integer :: rec_len_lon = 180*120
 integer :: rec_len, length
 integer :: irec, lrec, nrec, ierr
 integer*1, allocatable, dimension(:) :: soil_int

! get the file name from command line input

!call getarg(1,soilfile)
!print *, 'opening file ', soilfile

! record length for the data
!    each record has 360x120 data points

 length = rec_len_lat/4

 length = length*4                      

! open old 30 sec direct access file and 30 sec sequential file to read:

 open (iunitin,  file='topsoil30snew',access='DIRECT',recl=length,status='old')

! open direct access file for merged 30 sec soil data: 

!open (iunit,file=soilfile,access='DIRECT',recl=length,status='NEW')

! nrec: from north to south (max 21600)
! irec: from west to east   (max 43200)

 ierr = 0
 allocate (soil_int (rec_len_lat))
 allocate (soil_char(rec_len_lat))

 do nrec = 1, rec_len_lon

    read (iunitin,rec=nrec) soil_int 

!   if(mod(nrec,120) == 0) print *, 'nrec,soilcat = ', nrec, soil_int(6600)
    if(mod(nrec,120) == 0) print '(i5,i5)',nrec,soil_int(10800)

!   soil_char = achar(soil_int)
!   write (iunit, rec=nrec) soil_char

    if (ierr /= 0) then
       print *, 'read error on record nrec = ', nrec
       stop 'exit read error'
    endif

 end do

 deallocate (soil_char)
 deallocate (soil_int )

 stop
end program read_soil

