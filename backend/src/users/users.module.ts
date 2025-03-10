import { Module } from '@nestjs/common';
import { UsersService } from './users.service';
import { UsersV1Controller } from './users.controller';

@Module({
  controllers: [UsersV1Controller],
  providers: [UsersService],
  exports: [UsersService],
})
export class UsersModule {}
