import { Injectable, NotFoundException } from '@nestjs/common';
import { CreateUserDto } from './dto/create-user.dto';
import { UpdateUserDto } from './dto/update-user.dto';
import { PrismaService } from 'nestjs-prisma';

@Injectable()
export class UsersService {
  constructor(private readonly prismaService: PrismaService) {}

  create(createUserDto: CreateUserDto) {
    console.log(createUserDto);
    return 'This action adds a new user';
  }

  findAll() {
    return `This action returns all users`;
  }

  async getUserById(id: string) {
    try {
      const user = await this.prismaService.user.findUnique({
        where: {
          id: id,
        },
      });

      if (!user) throw new NotFoundException('User not found');

      return user;
    } catch (error) {
      console.log('Error in findOne method', error);
      throw error;
    }
  }

  async getUserByEmail(email: string) {
    try {
      const user = await this.prismaService.user.findUnique({
        where: {
          email: email,
        },
      });

      if (!user) throw new NotFoundException('User not found');

      return user;
    } catch (error) {
      console.log('Error in findOne method', error);
      throw error;
    }
  }

  update(id: number, updateUserDto: UpdateUserDto) {
    console.log(updateUserDto);
    return `This action updates a #${id} user`;
  }

  remove(id: number) {
    return `This action removes a #${id} user`;
  }
}
